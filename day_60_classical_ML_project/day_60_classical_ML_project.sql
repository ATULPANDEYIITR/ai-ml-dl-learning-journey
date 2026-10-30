-- Classical ML experiment registry and evaluation workflow.
-- Compatible with PostgreSQL 14+.
--
-- The script creates a relational experiment registry, feature catalogue,
-- data-quality records, preprocessing lineage, model evaluations, and
-- deployment decisions. Seed data represents a curated six-feature
-- Wisconsin diagnostic classification experiment.

DROP SCHEMA IF EXISTS classical_ml CASCADE;
CREATE SCHEMA classical_ml;
SET search_path TO classical_ml, public;

CREATE TYPE dataset_state AS ENUM (
    'draft',
    'validated',
    'quarantined',
    'retired'
);

CREATE TYPE experiment_state AS ENUM (
    'created',
    'validated',
    'running',
    'completed',
    'failed'
);

CREATE TYPE model_run_state AS ENUM (
    'registered',
    'training',
    'completed',
    'failed'
);

CREATE TYPE evaluation_split AS ENUM (
    'holdout_test',
    'cross_validation'
);

-- A dataset version identifies the exact data snapshot used by an experiment.
CREATE TABLE dataset_versions (
    dataset_id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_key             TEXT NOT NULL,
    version                 TEXT NOT NULL,
    source_description      TEXT NOT NULL,
    row_count               INTEGER NOT NULL CHECK (row_count > 0),
    feature_count           INTEGER NOT NULL CHECK (feature_count > 0),
    target_column           TEXT NOT NULL,
    positive_class          TEXT NOT NULL,
    status                  dataset_state NOT NULL DEFAULT 'draft',
    checksum_sha256         TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (dataset_key, version),
    CHECK (
        checksum_sha256 IS NULL OR
        checksum_sha256 ~ '^[0-9a-f]{64}$'
    )
);

CREATE TABLE feature_catalog (
    feature_id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id              BIGINT NOT NULL
        REFERENCES dataset_versions(dataset_id) ON DELETE CASCADE,
    feature_name            TEXT NOT NULL,
    data_type               TEXT NOT NULL,
    feature_role            TEXT NOT NULL DEFAULT 'predictor'
        CHECK (feature_role IN ('predictor', 'identifier', 'target')),
    feature_order           INTEGER NOT NULL CHECK (feature_order >= 0),
    missing_count           INTEGER NOT NULL DEFAULT 0
        CHECK (missing_count >= 0),
    selected_for_model      BOOLEAN NOT NULL DEFAULT TRUE,
    engineering_expression TEXT,
    UNIQUE (dataset_id, feature_name),
    UNIQUE (dataset_id, feature_order)
);

CREATE TABLE data_quality_checks (
    quality_check_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id              BIGINT NOT NULL
        REFERENCES dataset_versions(dataset_id) ON DELETE CASCADE,
    check_name              TEXT NOT NULL,
    passed                  BOOLEAN NOT NULL,
    details                 JSONB NOT NULL DEFAULT '{}'::JSONB,
    checked_at              TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (dataset_id, check_name)
);

CREATE TABLE experiments (
    experiment_id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id              BIGINT NOT NULL
        REFERENCES dataset_versions(dataset_id),
    experiment_key          TEXT NOT NULL UNIQUE,
    algorithm               TEXT NOT NULL,
    algorithm_version       TEXT NOT NULL,
    random_seed             INTEGER NOT NULL CHECK (random_seed >= 0),
    train_fraction          NUMERIC(5,4) NOT NULL,
    validation_fraction     NUMERIC(5,4) NOT NULL DEFAULT 0,
    test_fraction           NUMERIC(5,4) NOT NULL,
    parameters              JSONB NOT NULL DEFAULT '{}'::JSONB,
    state                   experiment_state NOT NULL DEFAULT 'created',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at            TIMESTAMPTZ,
    CHECK (
        train_fraction > 0 AND
        validation_fraction >= 0 AND
        test_fraction > 0 AND
        train_fraction + validation_fraction + test_fraction = 1
    ),
    CHECK (
        (state = 'completed' AND completed_at IS NOT NULL)
        OR
        (state <> 'completed')
    )
);

-- Sample counts and fingerprints make the split auditable.
CREATE TABLE data_splits (
    split_id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id           BIGINT NOT NULL
        REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    split_name              TEXT NOT NULL
        CHECK (split_name IN ('train', 'validation', 'holdout_test')),
    sample_count            INTEGER NOT NULL CHECK (sample_count > 0),
    sample_fingerprint      TEXT NOT NULL,
    stratified              BOOLEAN NOT NULL DEFAULT TRUE,
    UNIQUE (experiment_id, split_name)
);

-- fit_scope distinguishes row-level transformations from learned parameters.
-- Learned imputation, scaling, and selection must use training data only.
CREATE TABLE preprocessing_steps (
    preprocessing_step_id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id           BIGINT NOT NULL
        REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    step_order              INTEGER NOT NULL CHECK (step_order > 0),
    step_type               TEXT NOT NULL
        CHECK (
            step_type IN (
                'imputation',
                'scaling',
                'encoding',
                'feature_selection',
                'feature_engineering'
            )
        ),
    configuration           JSONB NOT NULL DEFAULT '{}'::JSONB,
    fit_scope               TEXT NOT NULL
        CHECK (fit_scope IN ('none', 'train_only', 'all_data')),
    fitted                  BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (experiment_id, step_order)
);

CREATE TABLE model_runs (
    model_run_id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id           BIGINT NOT NULL
        REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    model_name              TEXT NOT NULL,
    model_version           TEXT NOT NULL,
    status                  model_run_state NOT NULL DEFAULT 'registered',
    artifact_uri            TEXT,
    training_started_at     TIMESTAMPTZ,
    training_completed_at   TIMESTAMPTZ,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (experiment_id, model_name, model_version),
    CHECK (
        (status = 'completed' AND training_completed_at IS NOT NULL)
        OR
        (status <> 'completed')
    )
);

CREATE TABLE model_evaluations (
    evaluation_id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_run_id            BIGINT NOT NULL
        REFERENCES model_runs(model_run_id) ON DELETE CASCADE,
    split_type              evaluation_split NOT NULL,
    accuracy                NUMERIC(8,7) NOT NULL
        CHECK (accuracy BETWEEN 0 AND 1),
    precision_score         NUMERIC(8,7) NOT NULL
        CHECK (precision_score BETWEEN 0 AND 1),
    recall_score            NUMERIC(8,7) NOT NULL
        CHECK (recall_score BETWEEN 0 AND 1),
    f1_score                NUMERIC(8,7) NOT NULL
        CHECK (f1_score BETWEEN 0 AND 1),
    roc_auc                 NUMERIC(8,7) NOT NULL
        CHECK (roc_auc BETWEEN 0 AND 1),
    cv_f1_mean              NUMERIC(8,7),
    cv_f1_std               NUMERIC(8,7),
    true_negative           INTEGER,
    false_positive          INTEGER,
    false_negative          INTEGER,
    true_positive           INTEGER,
    evaluated_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (model_run_id, split_type),
    CHECK (
        (
            split_type = 'holdout_test'
            AND cv_f1_mean IS NULL
            AND cv_f1_std IS NULL
            AND true_negative IS NOT NULL
            AND false_positive IS NOT NULL
            AND false_negative IS NOT NULL
            AND true_positive IS NOT NULL
        )
        OR
        (
            split_type = 'cross_validation'
            AND cv_f1_mean BETWEEN 0 AND 1
            AND cv_f1_std BETWEEN 0 AND 1
            AND true_negative IS NULL
            AND false_positive IS NULL
            AND false_negative IS NULL
            AND true_positive IS NULL
        )
    ),
    CHECK (
        COALESCE(true_negative, 0) >= 0 AND
        COALESCE(false_positive, 0) >= 0 AND
        COALESCE(false_negative, 0) >= 0 AND
        COALESCE(true_positive, 0) >= 0
    )
);

CREATE TABLE deployment_policies (
    policy_id                   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    policy_name                 TEXT NOT NULL UNIQUE,
    minimum_recall              NUMERIC(5,4) NOT NULL
        CHECK (minimum_recall BETWEEN 0 AND 1),
    minimum_precision           NUMERIC(5,4) NOT NULL
        CHECK (minimum_precision BETWEEN 0 AND 1),
    minimum_roc_auc             NUMERIC(5,4) NOT NULL
        CHECK (minimum_roc_auc BETWEEN 0 AND 1),
    minimum_cv_f1               NUMERIC(5,4) NOT NULL
        CHECK (minimum_cv_f1 BETWEEN 0 AND 1),
    maximum_cv_f1_std           NUMERIC(5,4) NOT NULL
        CHECK (maximum_cv_f1_std BETWEEN 0 AND 1),
    require_passed_quality      BOOLEAN NOT NULL DEFAULT TRUE,
    require_train_only_fit      BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE deployment_decisions (
    decision_id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_run_id            BIGINT NOT NULL
        REFERENCES model_runs(model_run_id),
    policy_id               BIGINT NOT NULL
        REFERENCES deployment_policies(policy_id),
    decision_status         TEXT NOT NULL
        CHECK (decision_status IN ('eligible', 'blocked')),
    reasons                 TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    decided_at              TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes support the main registry and reporting access patterns.
CREATE INDEX idx_experiments_dataset_state
    ON experiments(dataset_id, state);

CREATE INDEX idx_model_runs_experiment_status
    ON model_runs(experiment_id, status);

CREATE INDEX idx_evaluations_run_split
    ON model_evaluations(model_run_id, split_type);

CREATE INDEX idx_quality_checks_dataset_passed
    ON data_quality_checks(dataset_id, passed);

CREATE INDEX idx_decisions_run_time
    ON deployment_decisions(model_run_id, decided_at DESC);

-- Permit only valid experiment lifecycle transitions.
CREATE FUNCTION guard_experiment_transition()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.state = OLD.state THEN
        RETURN NEW;
    END IF;

    IF NOT (
        (OLD.state = 'created' AND NEW.state IN ('validated', 'failed'))
        OR
        (OLD.state = 'validated' AND NEW.state IN ('running', 'failed'))
        OR
        (OLD.state = 'running' AND NEW.state IN ('completed', 'failed'))
    ) THEN
        RAISE EXCEPTION
            'Invalid experiment state transition: % -> %',
            OLD.state,
            NEW.state
            USING ERRCODE = 'check_violation';
    END IF;

    IF NEW.state = 'completed' AND NEW.completed_at IS NULL THEN
        NEW.completed_at = CURRENT_TIMESTAMP;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_guard_experiment_transition
BEFORE UPDATE OF state ON experiments
FOR EACH ROW
EXECUTE FUNCTION guard_experiment_transition();

-- Report combines holdout performance with cross-validation stability.
CREATE VIEW v_model_scorecard AS
SELECT
    mr.model_run_id,
    e.experiment_key,
    dv.dataset_key,
    dv.version AS dataset_version,
    mr.model_name,
    mr.model_version,
    mr.status AS model_status,
    test_eval.accuracy AS test_accuracy,
    test_eval.precision_score AS test_precision,
    test_eval.recall_score AS test_recall,
    test_eval.f1_score AS test_f1,
    test_eval.roc_auc AS test_roc_auc,
    cv_eval.cv_f1_mean,
    cv_eval.cv_f1_std
FROM model_runs AS mr
JOIN experiments AS e
    ON e.experiment_id = mr.experiment_id
JOIN dataset_versions AS dv
    ON dv.dataset_id = e.dataset_id
LEFT JOIN model_evaluations AS test_eval
    ON test_eval.model_run_id = mr.model_run_id
   AND test_eval.split_type = 'holdout_test'
LEFT JOIN model_evaluations AS cv_eval
    ON cv_eval.model_run_id = mr.model_run_id
   AND cv_eval.split_type = 'cross_validation';

-- Evaluate one model version against a deployment policy.
-- This function records its decision and returns the decision with reasons.
CREATE FUNCTION evaluate_deployment(
    p_model_run_id BIGINT,
    p_policy_id BIGINT
)
RETURNS TABLE (
    decision_status TEXT,
    reasons TEXT[]
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_dataset_status dataset_state;
    v_experiment_status experiment_state;
    v_model_status model_run_state;
    v_experiment_id BIGINT;
    v_dataset_id BIGINT;

    v_policy deployment_policies%ROWTYPE;
    v_test model_evaluations%ROWTYPE;
    v_cv model_evaluations%ROWTYPE;

    v_quality_total BIGINT;
    v_quality_failed BIGINT;

    v_reasons TEXT[] := ARRAY[]::TEXT[];
    v_decision TEXT;
BEGIN
    SELECT
        dv.status,
        e.state,
        mr.status,
        e.experiment_id,
        dv.dataset_id
    INTO
        v_dataset_status,
        v_experiment_status,
        v_model_status,
        v_experiment_id,
        v_dataset_id
    FROM model_runs AS mr
    JOIN experiments AS e
        ON e.experiment_id = mr.experiment_id
    JOIN dataset_versions AS dv
        ON dv.dataset_id = e.dataset_id
    WHERE mr.model_run_id = p_model_run_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Unknown model run: %', p_model_run_id;
    END IF;

    SELECT *
    INTO v_policy
    FROM deployment_policies
    WHERE policy_id = p_policy_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Unknown deployment policy: %', p_policy_id;
    END IF;

    IF v_dataset_status <> 'validated' THEN
        v_reasons := array_append(
            v_reasons,
            'Dataset is not in validated state.'
        );
    END IF;

    IF v_experiment_status <> 'completed' THEN
        v_reasons := array_append(
            v_reasons,
            'Experiment has not completed successfully.'
        );
    END IF;

    IF v_model_status <> 'completed' THEN
        v_reasons := array_append(
            v_reasons,
            'Model training has not completed successfully.'
        );
    END IF;

    SELECT
        COUNT(*),
        COUNT(*) FILTER (WHERE NOT passed)
    INTO
        v_quality_total,
        v_quality_failed
    FROM data_quality_checks
    WHERE dataset_id = v_dataset_id;

    IF v_policy.require_passed_quality THEN
        IF v_quality_total = 0 OR v_quality_failed > 0 THEN
            v_reasons := array_append(
                v_reasons,
                'Dataset quality checks are absent or contain failures.'
            );
        END IF;
    END IF;

    SELECT *
    INTO v_test
    FROM model_evaluations
    WHERE model_run_id = p_model_run_id
      AND split_type = 'holdout_test';

    IF NOT FOUND THEN
        v_reasons := array_append(
            v_reasons,
            'A final holdout test evaluation is missing.'
        );
    ELSE
        IF v_test.recall_score < v_policy.minimum_recall THEN
            v_reasons := array_append(
                v_reasons,
                'Holdout recall is below the policy threshold.'
            );
        END IF;

        IF v_test.precision_score < v_policy.minimum_precision THEN
            v_reasons := array_append(
                v_reasons,
                'Holdout precision is below the policy threshold.'
            );
        END IF;

        IF v_test.roc_auc < v_policy.minimum_roc_auc THEN
            v_reasons := array_append(
                v_reasons,
                'Holdout ROC AUC is below the policy threshold.'
            );
        END IF;
    END IF;

    SELECT *
    INTO v_cv
    FROM model_evaluations
    WHERE model_run_id = p_model_run_id
      AND split_type = 'cross_validation';

    IF NOT FOUND THEN
        v_reasons := array_append(
            v_reasons,
            'Cross-validation results are missing.'
        );
    ELSE
        IF v_cv.cv_f1_mean < v_policy.minimum_cv_f1 THEN
            v_reasons := array_append(
                v_reasons,
                'Mean cross-validation F1 is below the policy threshold.'
            );
        END IF;

        IF v_cv.cv_f1_std > v_policy.maximum_cv_f1_std THEN
            v_reasons := array_append(
                v_reasons,
                'Cross-validation F1 variation exceeds the policy limit.'
            );
        END IF;
    END IF;

    IF v_policy.require_train_only_fit THEN
        IF EXISTS (
            SELECT 1
            FROM preprocessing_steps
            WHERE experiment_id = v_experiment_id
              AND (
                    fit_scope = 'all_data'
                    OR (
                        step_type IN (
                            'imputation',
                            'scaling',
                            'encoding',
                            'feature_selection'
                        )
                        AND fit_scope <> 'train_only'
                    )
                    OR fitted = FALSE
                  )
        ) THEN
            v_reasons := array_append(
                v_reasons,
                'Learned preprocessing is unfitted or uses non-training data.'
            );
        END IF;
    END IF;

    v_decision := CASE
        WHEN cardinality(v_reasons) = 0 THEN 'eligible'
        ELSE 'blocked'
    END;

    INSERT INTO deployment_decisions (
        model_run_id,
        policy_id,
        decision_status,
        reasons
    )
    VALUES (
        p_model_run_id,
        p_policy_id,
        v_decision,
        v_reasons
    );

    RETURN QUERY
    SELECT v_decision, v_reasons;
END;
$$;

BEGIN;

-- Dataset version one is a curated six-feature diagnostic dataset.
INSERT INTO dataset_versions (
    dataset_key,
    version,
    source_description,
    row_count,
    feature_count,
    target_column,
    positive_class,
    status,
    checksum_sha256
)
VALUES
(
    'wisconsin_diagnostic_curated',
    '1.0',
    'Curated Wisconsin diagnostic classification snapshot',
    569,
    6,
    'target',
    'malignant',
    'validated',
    repeat('a', 64)
),
(
    'wisconsin_diagnostic_quarantined',
    '2.0',
    'Snapshot with an unresolved target-integrity quality failure',
    569,
    6,
    'target',
    'malignant',
    'quarantined',
    repeat('b', 64)
);

INSERT INTO feature_catalog (
    dataset_id,
    feature_name,
    data_type,
    feature_role,
    feature_order,
    missing_count,
    selected_for_model,
    engineering_expression
)
SELECT
    dv.dataset_id,
    feature.feature_name,
    'numeric',
    'predictor',
    feature.feature_order,
    0,
    TRUE,
    feature.engineering_expression
FROM dataset_versions AS dv
CROSS JOIN (
    VALUES
        ('mean_radius', 1, NULL::TEXT),
        ('mean_texture', 2, NULL::TEXT),
        ('mean_perimeter', 3, NULL::TEXT),
        ('mean_area', 4, NULL::TEXT),
        ('worst_radius', 5, NULL::TEXT),
        (
            'worst_to_mean_radius_ratio',
            6,
            'worst_radius / mean_radius'
        )
) AS feature(feature_name, feature_order, engineering_expression);

INSERT INTO data_quality_checks (
    dataset_id,
    check_name,
    passed,
    details
)
SELECT
    dataset_id,
    'unique_observation_identifiers',
    TRUE,
    '{"duplicate_count": 0}'::JSONB
FROM dataset_versions
WHERE dataset_key = 'wisconsin_diagnostic_curated';

INSERT INTO data_quality_checks (
    dataset_id,
    check_name,
    passed,
    details
)
SELECT
    dataset_id,
    'target_values_binary',
    TRUE,
    '{"unexpected_labels": 0, "classes_found": 2}'::JSONB
FROM dataset_versions
WHERE dataset_key = 'wisconsin_diagnostic_curated';

INSERT INTO data_quality_checks (
    dataset_id,
    check_name,
    passed,
    details
)
SELECT
    dataset_id,
    'feature_missingness_threshold',
    TRUE,
    '{"maximum_missing_fraction": 0.02, "configured_limit": 0.05}'::JSONB
FROM dataset_versions
WHERE dataset_key = 'wisconsin_diagnostic_curated';

INSERT INTO data_quality_checks (
    dataset_id,
    check_name,
    passed,
    details
)
SELECT
    dataset_id,
    'unique_observation_identifiers',
    TRUE,
    '{"duplicate_count": 0}'::JSONB
FROM dataset_versions
WHERE dataset_key = 'wisconsin_diagnostic_quarantined';

INSERT INTO data_quality_checks (
    dataset_id,
    check_name,
    passed,
    details
)
SELECT
    dataset_id,
    'target_values_binary',
    FALSE,
    '{"unexpected_labels": 7, "action": "quarantine"}'::JSONB
FROM dataset_versions
WHERE dataset_key = 'wisconsin_diagnostic_quarantined';

-- Experiments include a passing candidate, a weak candidate, and a run
-- against a quarantined snapshot to illustrate independent blocking rules.
INSERT INTO experiments (
    dataset_id,
    experiment_key,
    algorithm,
    algorithm_version,
    random_seed,
    train_fraction,
    validation_fraction,
    test_fraction,
    parameters,
    state,
    completed_at
)
SELECT
    dataset_id,
    'EXP-LOGISTIC-001',
    'LogisticRegression',
    '1.0',
    42,
    0.65,
    0.15,
    0.20,
    '{"C": 1.0, "class_weight": "balanced"}'::JSONB,
    'completed',
    CURRENT_TIMESTAMP
FROM dataset_versions
WHERE dataset_key = 'wisconsin_diagnostic_curated';

INSERT INTO experiments (
    dataset_id,
    experiment_key,
    algorithm,
    algorithm_version,
    random_seed,
    train_fraction,
    validation_fraction,
    test_fraction,
    parameters,
    state,
    completed_at
)
SELECT
    dataset_id,
    'EXP-RANDOM-FOREST-001',
    'RandomForestClassifier',
    '1.0',
    42,
    0.65,
    0.15,
    0.20,
    '{"n_estimators": 350, "min_samples_leaf": 2}'::JSONB,
    'completed',
    CURRENT_TIMESTAMP
FROM dataset_versions
WHERE dataset_key = 'wisconsin_diagnostic_curated';

INSERT INTO experiments (
    dataset_id,
    experiment_key,
    algorithm,
    algorithm_version,
    random_seed,
    train_fraction,
    validation_fraction,
    test_fraction,
    parameters,
    state,
    completed_at
)
SELECT
    dataset_id,
    'EXP-QUARANTINED-001',
    'LogisticRegression',
    '1.0',
    42,
    0.65,
    0.15,
    0.20,
    '{"C": 1.0}'::JSONB,
    'completed',
    CURRENT_TIMESTAMP
FROM dataset_versions
WHERE dataset_key = 'wisconsin_diagnostic_quarantined';

INSERT INTO data_splits (
    experiment_id,
    split_name,
    sample_count,
    sample_fingerprint,
    stratified
)
SELECT
    experiment_id,
    split.split_name,
    split.sample_count,
    split.fingerprint,
    TRUE
FROM experiments
CROSS JOIN (
    VALUES
        ('train', 370, 'train-fingerprint-455'),
        ('validation', 85, 'validation-fingerprint-455'),
        ('holdout_test', 114, 'test-fingerprint-455')
) AS split(split_name, sample_count, fingerprint);

INSERT INTO preprocessing_steps (
    experiment_id,
    step_order,
    step_type,
    configuration,
    fit_scope,
    fitted
)
SELECT
    experiment_id,
    steps.step_order,
    steps.step_type,
    steps.configuration,
    steps.fit_scope,
    TRUE
FROM experiments
CROSS JOIN (
    VALUES
        (
            1,
            'imputation',
            '{"strategy": "median"}'::JSONB,
            'train_only'
        ),
        (
            2,
            'feature_engineering',
            '{"operation": "worst_radius / mean_radius"}'::JSONB,
            'none'
        ),
        (
            3,
            'scaling',
            '{"strategy": "standard_scaler"}'::JSONB,
            'train_only'
        )
) AS steps(
    step_order,
    step_type,
    configuration,
    fit_scope
)
WHERE experiment_key <> 'EXP-QUARANTINED-001';

INSERT INTO preprocessing_steps (
    experiment_id,
    step_order,
    step_type,
    configuration,
    fit_scope,
    fitted
)
SELECT
    experiment_id,
    1,
    'scaling',
    '{"strategy": "standard_scaler"}'::JSONB,
    'all_data',
    TRUE
FROM experiments
WHERE experiment_key = 'EXP-QUARANTINED-001';

INSERT INTO model_runs (
    experiment_id,
    model_name,
    model_version,
    status,
    artifact_uri,
    training_started_at,
    training_completed_at
)
SELECT
    experiment_id,
    experiment_key,
    '1.0',
    'completed',
    'model-registry://wisconsin/logistic/1.0',
    CURRENT_TIMESTAMP - INTERVAL '10 minutes',
    CURRENT_TIMESTAMP
FROM experiments
WHERE experiment_key = 'EXP-LOGISTIC-001';

INSERT INTO model_runs (
    experiment_id,
    model_name,
    model_version,
    status,
    artifact_uri,
    training_started_at,
    training_completed_at
)
SELECT
    experiment_id,
    experiment_key,
    '1.0',
    'completed',
    'model-registry://wisconsin/random-forest/1.0',
    CURRENT_TIMESTAMP - INTERVAL '9 minutes',
    CURRENT_TIMESTAMP
FROM experiments
WHERE experiment_key = 'EXP-RANDOM-FOREST-001';

INSERT INTO model_runs (
    experiment_id,
    model_name,
    model_version,
    status,
    artifact_uri,
    training_started_at,
    training_completed_at
)
SELECT
    experiment_id,
    experiment_key,
    '1.0',
    'completed',
    'model-registry://wisconsin/quarantined/1.0',
    CURRENT_TIMESTAMP - INTERVAL '8 minutes',
    CURRENT_TIMESTAMP
FROM experiments
WHERE experiment_key = 'EXP-QUARANTINED-001';

-- Logistic regression: holdout test and cross-validation results.
INSERT INTO model_evaluations (
    model_run_id,
    split_type,
    accuracy,
    precision_score,
    recall_score,
    f1_score,
    roc_auc,
    true_negative,
    false_positive,
    false_negative,
    true_positive
)
SELECT
    model_run_id,
    'holdout_test',
    0.9561404,
    0.9512195,
    0.9285714,
    0.9397590,
    0.9870000,
    70,
    2,
    3,
    39
FROM model_runs
WHERE model_name = 'EXP-LOGISTIC-001';

INSERT INTO model_evaluations (
    model_run_id,
    split_type,
    accuracy,
    precision_score,
    recall_score,
    f1_score,
    roc_auc,
    cv_f1_mean,
    cv_f1_std
)
SELECT
    model_run_id,
    'cross_validation',
    0.9400000,
    0.9350000,
    0.9330000,
    0.9340000,
    0.9820000,
    0.9340000,
    0.0180000
FROM model_runs
WHERE model_name = 'EXP-LOGISTIC-001';

-- Random forest: high precision but insufficient malignant recall.
INSERT INTO model_evaluations (
    model_run_id,
    split_type,
    accuracy,
    precision_score,
    recall_score,
    f1_score,
    roc_auc,
    true_negative,
    false_positive,
    false_negative,
    true_positive
)
SELECT
    model_run_id,
    'holdout_test',
    0.8859649,
    0.8918919,
    0.7857143,
    0.8354430,
    0.9250000,
    68,
    4,
    9,
    33
FROM model_runs
WHERE model_name = 'EXP-RANDOM-FOREST-001';

INSERT INTO model_evaluations (
    model_run_id,
    split_type,
    accuracy,
    precision_score,
    recall_score,
    f1_score,
    roc_auc,
    cv_f1_mean,
    cv_f1_std
)
SELECT
    model_run_id,
    'cross_validation',
    0.8900000,
    0.8800000,
    0.8400000,
    0.8600000,
    0.9300000,
    0.8600000,
    0.0400000
FROM model_runs
WHERE model_name = 'EXP-RANDOM-FOREST-001';

-- This run has strong metrics but fails dataset-quality and preprocessing checks.
INSERT INTO model_evaluations (
    model_run_id,
    split_type,
    accuracy,
    precision_score,
    recall_score,
    f1_score,
    roc_auc,
    true_negative,
    false_positive,
    false_negative,
    true_positive
)
SELECT
    model_run_id,
    'holdout_test',
    0.9700000,
    0.9600000,
    0.9500000,
    0.9550000,
    0.9900000,
    71,
    1,
    2,
    40
FROM model_runs
WHERE model_name = 'EXP-QUARANTINED-001';

INSERT INTO model_evaluations (
    model_run_id,
    split_type,
    accuracy,
    precision_score,
    recall_score,
    f1_score,
    roc_auc,
    cv_f1_mean,
    cv_f1_std
)
SELECT
    model_run_id,
    'cross_validation',
    0.9600000,
    0.9550000,
    0.9500000,
    0.9520000,
    0.9890000,
    0.9520000,
    0.0150000
FROM model_runs
WHERE model_name = 'EXP-QUARANTINED-001';

INSERT INTO deployment_policies (
    policy_name,
    minimum_recall,
    minimum_precision,
    minimum_roc_auc,
    minimum_cv_f1,
    maximum_cv_f1_std,
    require_passed_quality,
    require_train_only_fit
)
VALUES (
    'Malignant Classification Quality Gate',
    0.9000,
    0.8500,
    0.9500,
    0.9000,
    0.0500,
    TRUE,
    TRUE
);

COMMIT;

-- Invalid terminal-state transition is caught and reported without
-- stopping the remainder of the script.
DO $$
BEGIN
    BEGIN
        UPDATE experiments
        SET state = 'running'
        WHERE experiment_key = 'EXP-LOGISTIC-001';

    EXCEPTION WHEN check_violation THEN
        RAISE NOTICE 'Expected state-transition rejection: %', SQLERRM;
    END;
END;
$$;

-- Retrieval: model versions and their associated dataset versions.
SELECT
    model_run_id,
    experiment_key,
    dataset_key,
    dataset_version,
    model_name,
    model_status
FROM v_model_scorecard
ORDER BY model_run_id;

-- Validation: dataset checks and any failing records.
SELECT
    dv.dataset_key,
    dv.version,
    qc.check_name,
    qc.passed,
    qc.details
FROM dataset_versions AS dv
JOIN data_quality_checks AS qc
    ON qc.dataset_id = dv.dataset_id
ORDER BY dv.dataset_id, qc.quality_check_id;

-- Aggregation: model performance and stability by algorithm.
SELECT
    model_name,
    ROUND(AVG(test_f1), 4) AS average_test_f1,
    ROUND(AVG(test_recall), 4) AS average_test_recall,
    ROUND(AVG(test_roc_auc), 4) AS average_test_auc,
    ROUND(AVG(cv_f1_mean), 4) AS average_cv_f1,
    ROUND(AVG(cv_f1_std), 4) AS average_cv_f1_std
FROM v_model_scorecard
GROUP BY model_name
ORDER BY average_test_f1 DESC;

-- CTE ranking: compare holdout quality while considering validation stability.
WITH candidate_scores AS (
    SELECT
        model_run_id,
        model_name,
        test_recall,
        test_precision,
        test_roc_auc,
        test_f1,
        cv_f1_mean,
        cv_f1_std,
        (
            0.40 * test_recall +
            0.30 * test_roc_auc +
            0.20 * test_precision +
            0.10 * cv_f1_mean
        ) AS weighted_score
    FROM v_model_scorecard
    WHERE test_f1 IS NOT NULL
)
SELECT
    *,
    RANK() OVER (ORDER BY weighted_score DESC) AS candidate_rank
FROM candidate_scores
ORDER BY candidate_rank, model_run_id;

-- Transactionally evaluate all three model versions against the policy.
BEGIN;

SELECT
    mr.model_run_id,
    result.decision_status,
    result.reasons
FROM model_runs AS mr
CROSS JOIN LATERAL evaluate_deployment(
    mr.model_run_id,
    (
        SELECT policy_id
        FROM deployment_policies
        WHERE policy_name = 'Malignant Classification Quality Gate'
    )
) AS result
ORDER BY mr.model_run_id;

COMMIT;

-- Reporting: latest eligibility decisions and their blocking reasons.
SELECT
    mr.model_name,
    dp.policy_name,
    dd.decision_status,
    dd.reasons,
    dd.decided_at
FROM deployment_decisions AS dd
JOIN model_runs AS mr
    ON mr.model_run_id = dd.model_run_id
JOIN deployment_policies AS dp
    ON dp.policy_id = dd.policy_id
ORDER BY dd.decided_at DESC, dd.decision_id DESC;

-- Workflow state: count experiments and models by lifecycle state.
SELECT
    e.state AS experiment_state,
    mr.status AS model_state,
    COUNT(*) AS model_count
FROM experiments AS e
JOIN model_runs AS mr
    ON mr.experiment_id = e.experiment_id
GROUP BY e.state, mr.status
ORDER BY e.state, mr.status;

-- Feature catalogue: inspect feature definitions and engineering lineage.
SELECT
    dv.dataset_key,
    fc.feature_order,
    fc.feature_name,
    fc.data_type,
    fc.feature_role,
    fc.missing_count,
    fc.selected_for_model,
    fc.engineering_expression
FROM feature_catalog AS fc
JOIN dataset_versions AS dv
    ON dv.dataset_id = fc.dataset_id
ORDER BY dv.dataset_id, fc.feature_order;
