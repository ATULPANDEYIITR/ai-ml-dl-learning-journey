/*
    IMBALANCED LEARNING
    PostgreSQL 15+

    Domain: Financial transaction fraud detection.

    Models:
    - Datasets and observations
    - Class distribution
    - Imbalance assessment
    - Sampling strategies
    - Resampling operations
    - Synthetic SMOTE observations
    - Class weights
    - Model experiments
    - Predictions
    - Evaluation metrics
    - Experiment eligibility and reporting

    Sampling is restricted to training datasets.
*/

BEGIN;

CREATE SCHEMA IF NOT EXISTS imbalanced_learning;

SET search_path TO imbalanced_learning, public;

-- =========================================================
-- 1. DOMAIN TABLES
-- =========================================================

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_name VARCHAR(150) NOT NULL,
    dataset_version VARCHAR(40) NOT NULL,
    dataset_role VARCHAR(20) NOT NULL
        CHECK (
            dataset_role IN (
                'training',
                'validation',
                'testing'
            )
        ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    random_seed INTEGER NOT NULL DEFAULT 42,
    is_locked BOOLEAN NOT NULL DEFAULT FALSE,

    UNIQUE(dataset_name, dataset_version, dataset_role)
);

CREATE TABLE IF NOT EXISTS observations (
    observation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL
        REFERENCES datasets(dataset_id),
    source_observation_id BIGINT
        REFERENCES observations(observation_id),
    customer_reference VARCHAR(100) NOT NULL,
    transaction_amount NUMERIC(14,2) NOT NULL
        CHECK (transaction_amount >= 0),
    transaction_velocity NUMERIC(12,4) NOT NULL
        CHECK (transaction_velocity >= 0),
    account_age_days INTEGER NOT NULL
        CHECK (account_age_days >= 0),
    label SMALLINT NOT NULL
        CHECK (label IN (0, 1)),
    is_synthetic BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(dataset_id, customer_reference)
);

CREATE TABLE IF NOT EXISTS sampling_strategies (
    strategy_id SMALLINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    strategy_code VARCHAR(40) NOT NULL UNIQUE,
    strategy_name VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    requires_training_data BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS resampling_runs (
    resampling_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_dataset_id BIGINT NOT NULL
        REFERENCES datasets(dataset_id),
    output_dataset_id BIGINT NOT NULL
        REFERENCES datasets(dataset_id),
    strategy_id SMALLINT NOT NULL
        REFERENCES sampling_strategies(strategy_id),
    random_seed INTEGER NOT NULL DEFAULT 42,
    neighbors INTEGER,
    generated_count INTEGER NOT NULL DEFAULT 0
        CHECK (generated_count >= 0),
    executed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status VARCHAR(20) NOT NULL DEFAULT 'completed'
        CHECK (
            status IN ('pending', 'completed', 'failed')
        ),

    CHECK (source_dataset_id <> output_dataset_id),
    CHECK (neighbors IS NULL OR neighbors > 0)
);

CREATE TABLE IF NOT EXISTS class_weights (
    weight_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL
        REFERENCES datasets(dataset_id),
    label SMALLINT NOT NULL
        CHECK (label IN (0, 1)),
    sample_count INTEGER NOT NULL
        CHECK (sample_count > 0),
    class_weight NUMERIC(12,6) NOT NULL
        CHECK (class_weight > 0),
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(dataset_id, label)
);

CREATE TABLE IF NOT EXISTS model_experiments (
    experiment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_name VARCHAR(150) NOT NULL,
    training_dataset_id BIGINT NOT NULL
        REFERENCES datasets(dataset_id),
    testing_dataset_id BIGINT NOT NULL
        REFERENCES datasets(dataset_id),
    strategy_id SMALLINT NOT NULL
        REFERENCES sampling_strategies(strategy_id),
    model_name VARCHAR(100) NOT NULL,
    decision_threshold NUMERIC(6,5) NOT NULL DEFAULT 0.50000
        CHECK (
            decision_threshold > 0
            AND decision_threshold < 1
        ),
    status VARCHAR(20) NOT NULL DEFAULT 'created'
        CHECK (
            status IN (
                'created',
                'training',
                'completed',
                'failed'
            )
        ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CHECK (training_dataset_id <> testing_dataset_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id BIGINT NOT NULL
        REFERENCES model_experiments(experiment_id),
    observation_id BIGINT NOT NULL
        REFERENCES observations(observation_id),
    actual_label SMALLINT NOT NULL
        CHECK (actual_label IN (0, 1)),
    predicted_label SMALLINT NOT NULL
        CHECK (predicted_label IN (0, 1)),
    predicted_probability NUMERIC(8,7) NOT NULL
        CHECK (
            predicted_probability >= 0
            AND predicted_probability <= 1
        ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(experiment_id, observation_id)
);

CREATE TABLE IF NOT EXISTS evaluation_metrics (
    metric_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id BIGINT NOT NULL UNIQUE
        REFERENCES model_experiments(experiment_id),
    true_positive INTEGER NOT NULL CHECK (true_positive >= 0),
    true_negative INTEGER NOT NULL CHECK (true_negative >= 0),
    false_positive INTEGER NOT NULL CHECK (false_positive >= 0),
    false_negative INTEGER NOT NULL CHECK (false_negative >= 0),
    precision_score NUMERIC(10,7),
    recall_score NUMERIC(10,7),
    f1_score NUMERIC(10,7),
    specificity NUMERIC(10,7),
    balanced_accuracy NUMERIC(10,7),
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CHECK (
        precision_score IS NULL
        OR precision_score BETWEEN 0 AND 1
    ),
    CHECK (
        recall_score IS NULL
        OR recall_score BETWEEN 0 AND 1
    ),
    CHECK (
        f1_score IS NULL
        OR f1_score BETWEEN 0 AND 1
    ),
    CHECK (
        specificity IS NULL
        OR specificity BETWEEN 0 AND 1
    ),
    CHECK (
        balanced_accuracy IS NULL
        OR balanced_accuracy BETWEEN 0 AND 1
    )
);

-- =========================================================
-- 2. INDEXES
-- =========================================================

CREATE INDEX IF NOT EXISTS idx_observations_dataset_label
ON observations(dataset_id, label);

CREATE INDEX IF NOT EXISTS idx_observations_source
ON observations(source_observation_id)
WHERE source_observation_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_observations_customer
ON observations(customer_reference);

CREATE INDEX IF NOT EXISTS idx_experiments_status
ON model_experiments(status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_experiment_label
ON predictions(experiment_id, actual_label, predicted_label);

CREATE INDEX IF NOT EXISTS idx_resampling_strategy
ON resampling_runs(strategy_id, executed_at DESC);

-- =========================================================
-- 3. STRATEGY CONFIGURATION
-- =========================================================

INSERT INTO sampling_strategies (
    strategy_code,
    strategy_name,
    description
)
VALUES
(
    'baseline',
    'Baseline',
    'Original training distribution without resampling.'
),
(
    'random_oversampling',
    'Random Oversampling',
    'Duplicates existing minority-class observations.'
),
(
    'random_undersampling',
    'Random Undersampling',
    'Reduces majority-class observations.'
),
(
    'smote',
    'SMOTE',
    'Generates synthetic minority observations using interpolation.'
),
(
    'class_weighting',
    'Class Weighting',
    'Adjusts the contribution of each class during model training.'
)
ON CONFLICT (strategy_code) DO NOTHING;

-- =========================================================
-- 4. DATA INTEGRITY TRIGGER
-- =========================================================

CREATE OR REPLACE FUNCTION validate_observation_dataset()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_role VARCHAR(20);
    v_locked BOOLEAN;
    v_source_dataset BIGINT;
BEGIN
    SELECT dataset_role, is_locked
    INTO v_role, v_locked
    FROM datasets
    WHERE dataset_id = NEW.dataset_id
    FOR SHARE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Dataset does not exist.';
    END IF;

    IF v_locked THEN
        RAISE EXCEPTION 'Dataset is locked against modification.';
    END IF;

    IF NEW.is_synthetic AND v_role <> 'training' THEN
        RAISE EXCEPTION
            'Synthetic observations are allowed only in training datasets.';
    END IF;

    IF NEW.source_observation_id IS NOT NULL THEN
        SELECT dataset_id
        INTO v_source_dataset
        FROM observations
        WHERE observation_id = NEW.source_observation_id;

        IF NOT FOUND THEN
            RAISE EXCEPTION 'Source observation does not exist.';
        END IF;

        IF v_source_dataset = NEW.dataset_id THEN
            RAISE EXCEPTION
                'Source and generated observation must be distinct.';
        END IF;
    END IF;

    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_validate_observation
ON observations;

CREATE TRIGGER trg_validate_observation
BEFORE INSERT OR UPDATE ON observations
FOR EACH ROW
EXECUTE FUNCTION validate_observation_dataset();

-- =========================================================
-- 5. EXPERIMENT DATASET VALIDATION
-- =========================================================

CREATE OR REPLACE FUNCTION validate_experiment_datasets()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_training_role VARCHAR(20);
    v_testing_role VARCHAR(20);
BEGIN
    SELECT dataset_role
    INTO v_training_role
    FROM datasets
    WHERE dataset_id = NEW.training_dataset_id;

    SELECT dataset_role
    INTO v_testing_role
    FROM datasets
    WHERE dataset_id = NEW.testing_dataset_id;

    IF v_training_role <> 'training' THEN
        RAISE EXCEPTION
            'Training dataset must have training role.';
    END IF;

    IF v_testing_role <> 'testing' THEN
        RAISE EXCEPTION
            'Testing dataset must have testing role.';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM observations train
        JOIN observations test
          ON train.customer_reference = test.customer_reference
        WHERE train.dataset_id = NEW.training_dataset_id
          AND test.dataset_id = NEW.testing_dataset_id
    ) THEN
        RAISE EXCEPTION
            'Customer leakage detected between training and testing.';
    END IF;

    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_validate_experiment
ON model_experiments;

CREATE TRIGGER trg_validate_experiment
BEFORE INSERT OR UPDATE ON model_experiments
FOR EACH ROW
EXECUTE FUNCTION validate_experiment_datasets();

-- =========================================================
-- 6. CLASS DISTRIBUTION FUNCTION
-- =========================================================

CREATE OR REPLACE FUNCTION get_class_distribution(
    p_dataset_id BIGINT
)
RETURNS TABLE (
    negative_count BIGINT,
    positive_count BIGINT,
    imbalance_ratio NUMERIC
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        COUNT(*) FILTER (WHERE label = 0),
        COUNT(*) FILTER (WHERE label = 1),
        CASE
            WHEN COUNT(*) FILTER (WHERE label = 1) = 0
                OR COUNT(*) FILTER (WHERE label = 0) = 0
            THEN NULL
            ELSE
                GREATEST(
                    COUNT(*) FILTER (WHERE label = 0),
                    COUNT(*) FILTER (WHERE label = 1)
                )::NUMERIC
                /
                LEAST(
                    COUNT(*) FILTER (WHERE label = 0),
                    COUNT(*) FILTER (WHERE label = 1)
                )
        END
    FROM observations
    WHERE dataset_id = p_dataset_id;
$$;

-- =========================================================
-- 7. CLASS WEIGHT CALCULATION
-- =========================================================

CREATE OR REPLACE PROCEDURE calculate_class_weights(
    p_dataset_id BIGINT
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_total BIGINT;
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM datasets
        WHERE dataset_id = p_dataset_id
          AND dataset_role = 'training'
          AND NOT is_locked
    ) THEN
        RAISE EXCEPTION
            'An unlocked training dataset is required.';
    END IF;

    SELECT COUNT(*)
    INTO v_total
    FROM observations
    WHERE dataset_id = p_dataset_id;

    IF v_total = 0 THEN
        RAISE EXCEPTION 'Training dataset is empty.';
    END IF;

    IF (
        SELECT COUNT(DISTINCT label)
        FROM observations
        WHERE dataset_id = p_dataset_id
    ) <> 2 THEN
        RAISE EXCEPTION
            'Both classes must be represented.';
    END IF;

    INSERT INTO class_weights (
        dataset_id,
        label,
        sample_count,
        class_weight
    )
    SELECT
        p_dataset_id,
        label,
        COUNT(*)::INTEGER,
        ROUND(
            v_total::NUMERIC / (2 * COUNT(*)),
            6
        )
    FROM observations
    WHERE dataset_id = p_dataset_id
    GROUP BY label
    ON CONFLICT (dataset_id, label)
    DO UPDATE SET
        sample_count = EXCLUDED.sample_count,
        class_weight = EXCLUDED.class_weight,
        calculated_at = NOW();
END;
$$;

-- =========================================================
-- 8. EXPERIMENT METRIC CALCULATION
-- =========================================================

CREATE OR REPLACE PROCEDURE calculate_experiment_metrics(
    p_experiment_id BIGINT
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_tp INTEGER;
    v_tn INTEGER;
    v_fp INTEGER;
    v_fn INTEGER;
    v_precision NUMERIC;
    v_recall NUMERIC;
    v_specificity NUMERIC;
    v_f1 NUMERIC;
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM model_experiments
        WHERE experiment_id = p_experiment_id
          AND status = 'completed'
    ) THEN
        RAISE EXCEPTION
            'Experiment must be completed before evaluation.';
    END IF;

    SELECT
        COUNT(*) FILTER (
            WHERE actual_label = 1 AND predicted_label = 1
        )::INTEGER,
        COUNT(*) FILTER (
            WHERE actual_label = 0 AND predicted_label = 0
        )::INTEGER,
        COUNT(*) FILTER (
            WHERE actual_label = 0 AND predicted_label = 1
        )::INTEGER,
        COUNT(*) FILTER (
            WHERE actual_label = 1 AND predicted_label = 0
        )::INTEGER
    INTO v_tp, v_tn, v_fp, v_fn
    FROM predictions
    WHERE experiment_id = p_experiment_id;

    IF v_tp + v_tn + v_fp + v_fn = 0 THEN
        RAISE EXCEPTION 'No predictions available.';
    END IF;

    v_precision := CASE
        WHEN v_tp + v_fp = 0 THEN 0
        ELSE v_tp::NUMERIC / (v_tp + v_fp)
    END;

    v_recall := CASE
        WHEN v_tp + v_fn = 0 THEN 0
        ELSE v_tp::NUMERIC / (v_tp + v_fn)
    END;

    v_specificity := CASE
        WHEN v_tn + v_fp = 0 THEN 0
        ELSE v_tn::NUMERIC / (v_tn + v_fp)
    END;

    v_f1 := CASE
        WHEN v_precision + v_recall = 0 THEN 0
        ELSE
            2 * v_precision * v_recall
            / (v_precision + v_recall)
    END;

    INSERT INTO evaluation_metrics (
        experiment_id,
        true_positive,
        true_negative,
        false_positive,
        false_negative,
        precision_score,
        recall_score,
        f1_score,
        specificity,
        balanced_accuracy
    )
    VALUES (
        p_experiment_id,
        v_tp,
        v_tn,
        v_fp,
        v_fn,
        v_precision,
        v_recall,
        v_f1,
        v_specificity,
        (v_recall + v_specificity) / 2
    )
    ON CONFLICT (experiment_id)
    DO UPDATE SET
        true_positive = EXCLUDED.true_positive,
        true_negative = EXCLUDED.true_negative,
        false_positive = EXCLUDED.false_positive,
        false_negative = EXCLUDED.false_negative,
        precision_score = EXCLUDED.precision_score,
        recall_score = EXCLUDED.recall_score,
        f1_score = EXCLUDED.f1_score,
        specificity = EXCLUDED.specificity,
        balanced_accuracy = EXCLUDED.balanced_accuracy,
        evaluated_at = NOW();
END;
$$;

-- =========================================================
-- 9. SAMPLE DATA
-- =========================================================

INSERT INTO datasets (
    dataset_name,
    dataset_version,
    dataset_role,
    random_seed
)
VALUES
    ('fraud_transactions', 'v1', 'training', 42),
    ('fraud_transactions', 'v1', 'testing', 42),
    ('fraud_transactions', 'v1', 'validation', 42);

-- Obtain dataset identifiers dynamically.

CREATE TEMP TABLE dataset_ids AS
SELECT
    dataset_id,
    dataset_role
FROM datasets
WHERE dataset_name = 'fraud_transactions'
  AND dataset_version = 'v1';

-- Training data: 8 normal and 2 fraudulent transactions.

INSERT INTO observations (
    dataset_id,
    customer_reference,
    transaction_amount,
    transaction_velocity,
    account_age_days,
    label
)
SELECT
    d.dataset_id,
    'TRAIN_' || g.i,
    CASE WHEN g.i <= 8 THEN 500 + g.i * 25
         ELSE 9000 + g.i * 500 END,
    CASE WHEN g.i <= 8 THEN 1.0 + g.i * 0.2
         ELSE 12.0 + g.i END,
    CASE WHEN g.i <= 8 THEN 500 + g.i * 10
         ELSE 5 + g.i END,
    CASE WHEN g.i <= 8 THEN 0 ELSE 1 END
FROM dataset_ids d
CROSS JOIN generate_series(1, 10) AS g(i)
WHERE d.dataset_role = 'training';

-- Testing data preserves the original class distribution.

INSERT INTO observations (
    dataset_id,
    customer_reference,
    transaction_amount,
    transaction_velocity,
    account_age_days,
    label
)
SELECT
    d.dataset_id,
    'TEST_' || g.i,
    CASE WHEN g.i <= 4 THEN 600 + g.i * 20
         ELSE 10000 + g.i * 100 END,
    CASE WHEN g.i <= 4 THEN 1.5 + g.i * 0.1
         ELSE 14.0 + g.i END,
    CASE WHEN g.i <= 4 THEN 600 + g.i * 10
         ELSE 3 + g.i END,
    CASE WHEN g.i <= 4 THEN 0 ELSE 1 END
FROM dataset_ids d
CROSS JOIN generate_series(1, 5) AS g(i)
WHERE d.dataset_role = 'testing';

-- Validation data is not resampled.

INSERT INTO observations (
    dataset_id,
    customer_reference,
    transaction_amount,
    transaction_velocity,
    account_age_days,
    label
)
SELECT
    d.dataset_id,
    'VALID_' || g.i,
    CASE WHEN g.i <= 4 THEN 450 + g.i * 30
         ELSE 9500 + g.i * 400 END,
    CASE WHEN g.i <= 4 THEN 1.2 + g.i * 0.3
         ELSE 13.0 + g.i END,
    CASE WHEN g.i <= 4 THEN 450 + g.i * 10
         ELSE 2 + g.i END,
    CASE WHEN g.i <= 4 THEN 0 ELSE 1 END
FROM dataset_ids d
CROSS JOIN generate_series(1, 5) AS g(i)
WHERE d.dataset_role = 'validation';

-- =========================================================
-- 10. DISTRIBUTION REPORT
-- =========================================================

SELECT
    d.dataset_name,
    d.dataset_role,
    distribution.negative_count,
    distribution.positive_count,
    distribution.imbalance_ratio
FROM datasets d
CROSS JOIN LATERAL get_class_distribution(d.dataset_id)
    AS distribution
WHERE d.dataset_name = 'fraud_transactions'
ORDER BY d.dataset_role;

-- =========================================================
-- 11. CLASS WEIGHTS
-- =========================================================

CALL calculate_class_weights(
    (SELECT dataset_id FROM dataset_ids
     WHERE dataset_role = 'training')
);

SELECT
    d.dataset_role,
    w.label,
    w.sample_count,
    w.class_weight
FROM class_weights w
JOIN datasets d ON d.dataset_id = w.dataset_id
ORDER BY w.label;

-- =========================================================
-- 12. EXPERIMENT REGISTRATION
-- =========================================================

INSERT INTO model_experiments (
    experiment_name,
    training_dataset_id,
    testing_dataset_id,
    strategy_id,
    model_name,
    decision_threshold
)
SELECT
    'Fraud baseline experiment',
    train.dataset_id,
    test.dataset_id,
    strategy.strategy_id,
    'weighted_centroid',
    0.50000
FROM datasets train
CROSS JOIN datasets test
JOIN sampling_strategies strategy
    ON strategy.strategy_code = 'baseline'
WHERE train.dataset_role = 'training'
  AND test.dataset_role = 'testing'
  AND train.dataset_name = 'fraud_transactions'
  AND test.dataset_name = 'fraud_transactions';

-- =========================================================
-- 13. PREDICTION RECORDS
-- =========================================================

-- These records demonstrate evaluation storage.
-- In production, predictions are inserted by the model service.

INSERT INTO predictions (
    experiment_id,
    observation_id,
    actual_label,
    predicted_label,
    predicted_probability
)
SELECT
    e.experiment_id,
    o.observation_id,
    o.label,
    CASE
        WHEN o.label = 1 AND o.transaction_velocity > 10 THEN 1
        WHEN o.label = 0 AND o.transaction_velocity > 10 THEN 1
        ELSE 0
    END,
    CASE
        WHEN o.transaction_velocity > 10 THEN 0.85
        ELSE 0.20
    END
FROM model_experiments e
JOIN observations o
    ON o.dataset_id = e.testing_dataset_id
WHERE e.experiment_name = 'Fraud baseline experiment';

UPDATE model_experiments
SET status = 'completed'
WHERE experiment_name = 'Fraud baseline experiment';

CALL calculate_experiment_metrics(
    (
        SELECT experiment_id
        FROM model_experiments
        WHERE experiment_name = 'Fraud baseline experiment'
    )
);

-- =========================================================
-- 14. VIEWS
-- =========================================================

CREATE OR REPLACE VIEW dataset_distribution_report AS
SELECT
    d.dataset_id,
    d.dataset_name,
    d.dataset_version,
    d.dataset_role,
    COUNT(o.observation_id) AS total_records,
    COUNT(*) FILTER (WHERE o.label = 0) AS negative_records,
    COUNT(*) FILTER (WHERE o.label = 1) AS positive_records,
    CASE
        WHEN COUNT(*) FILTER (WHERE o.label = 1) = 0
            OR COUNT(*) FILTER (WHERE o.label = 0) = 0
        THEN NULL
        ELSE
            GREATEST(
                COUNT(*) FILTER (WHERE o.label = 0),
                COUNT(*) FILTER (WHERE o.label = 1)
            )::NUMERIC
            /
            LEAST(
                COUNT(*) FILTER (WHERE o.label = 0),
                COUNT(*) FILTER (WHERE o.label = 1)
            )
    END AS imbalance_ratio
FROM datasets d
LEFT JOIN observations o
    ON o.dataset_id = d.dataset_id
GROUP BY
    d.dataset_id,
    d.dataset_name,
    d.dataset_version,
    d.dataset_role;

CREATE OR REPLACE VIEW experiment_performance_report AS
SELECT
    e.experiment_id,
    e.experiment_name,
    e.model_name,
    s.strategy_code,
    e.status,
    m.true_positive,
    m.true_negative,
    m.false_positive,
    m.false_negative,
    m.precision_score,
    m.recall_score,
    m.f1_score,
    m.specificity,
    m.balanced_accuracy
FROM model_experiments e
JOIN sampling_strategies s
    ON s.strategy_id = e.strategy_id
LEFT JOIN evaluation_metrics m
    ON m.experiment_id = e.experiment_id;

-- =========================================================
-- 15. ELIGIBILITY REPORT
-- =========================================================

WITH training_distribution AS (
    SELECT
        dataset_id,
        COUNT(*) FILTER (WHERE label = 0) AS negatives,
        COUNT(*) FILTER (WHERE label = 1) AS positives
    FROM observations
    GROUP BY dataset_id
),
testing_distribution AS (
    SELECT
        dataset_id,
        COUNT(*) FILTER (WHERE label = 0) AS negatives,
        COUNT(*) FILTER (WHERE label = 1) AS positives
    FROM observations
    GROUP BY dataset_id
)
SELECT
    e.experiment_name,
    CASE
        WHEN e.status <> 'completed'
            THEN 'NOT_COMPLETED'
        WHEN COALESCE(t.negatives, 0) = 0
            OR COALESCE(t.positives, 0) = 0
            THEN 'INVALID_TRAINING_DISTRIBUTION'
        WHEN COALESCE(v.negatives, 0) = 0
            OR COALESCE(v.positives, 0) = 0
            THEN 'INVALID_TESTING_DISTRIBUTION'
        WHEN m.experiment_id IS NULL
            THEN 'EVALUATION_REQUIRED'
        ELSE 'EVALUATED'
    END AS evaluation_status
FROM model_experiments e
LEFT JOIN training_distribution t
    ON t.dataset_id = e.training_dataset_id
LEFT JOIN testing_distribution v
    ON v.dataset_id = e.testing_dataset_id
LEFT JOIN evaluation_metrics m
    ON m.experiment_id = e.experiment_id;

-- =========================================================
-- 16. RETRIEVAL AND AGGREGATION
-- =========================================================

-- Retrieve synthetic records separately.

SELECT
    observation_id,
    source_observation_id,
    customer_reference,
    label,
    transaction_amount,
    transaction_velocity
FROM observations
WHERE is_synthetic = TRUE;

-- Aggregate original and synthetic class distributions.

SELECT
    d.dataset_role,
    o.is_synthetic,
    o.label,
    COUNT(*) AS record_count,
    AVG(o.transaction_amount) AS average_amount,
    AVG(o.transaction_velocity) AS average_velocity
FROM observations o
JOIN datasets d
    ON d.dataset_id = o.dataset_id
GROUP BY
    d.dataset_role,
    o.is_synthetic,
    o.label
ORDER BY
    d.dataset_role,
    o.is_synthetic,
    o.label;

-- Retrieve evaluation performance.

SELECT *
FROM experiment_performance_report
ORDER BY experiment_id;

-- =========================================================
-- 17. TRANSACTION SAFETY EXAMPLE
-- =========================================================

-- An invalid synthetic record in a testing dataset is rejected.
-- The savepoint allows the surrounding transaction to continue.

SAVEPOINT before_invalid_sampling;

DO $$
DECLARE
    v_testing_id BIGINT;
BEGIN
    SELECT dataset_id
    INTO v_testing_id
    FROM datasets
    WHERE dataset_name = 'fraud_transactions'
      AND dataset_version = 'v1'
      AND dataset_role = 'testing';

    BEGIN
        INSERT INTO observations (
            dataset_id,
            customer_reference,
            transaction_amount,
            transaction_velocity,
            account_age_days,
            label,
            is_synthetic
        )
        VALUES (
            v_testing_id,
            'INVALID_SYNTHETIC_TEST',
            1000,
            12,
            10,
            1,
            TRUE
        );

    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Expected validation failure: %', SQLERRM;
    END;
END;
$$;

RELEASE SAVEPOINT before_invalid_sampling;

-- =========================================================
-- 18. FINAL DATA QUALITY CHECKS
-- =========================================================

SELECT
    dataset_id,
    COUNT(*) AS total_records,
    COUNT(DISTINCT customer_reference) AS unique_customers,
    COUNT(*) FILTER (WHERE label = 0) AS normal_transactions,
    COUNT(*) FILTER (WHERE label = 1) AS fraudulent_transactions,
    COUNT(*) FILTER (WHERE is_synthetic) AS synthetic_transactions
FROM observations
GROUP BY dataset_id
ORDER BY dataset_id;

COMMIT;
