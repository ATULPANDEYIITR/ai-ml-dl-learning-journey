-- PostgreSQL 15+
-- Ensemble learning experiment registry and out-of-sample evaluation.
--
-- This schema records ensemble architectures, their component models,
-- validation strategies, prediction provenance, and evaluation metrics.
-- It distinguishes training, validation, and test observations to make
-- data leakage visible and to support reproducible comparisons.

BEGIN;

DROP VIEW IF EXISTS ensemble_metric_comparison CASCADE;
DROP VIEW IF EXISTS ensemble_prediction_summary CASCADE;
DROP TABLE IF EXISTS evaluation_metrics CASCADE;
DROP TABLE IF EXISTS ensemble_predictions CASCADE;
DROP TABLE IF EXISTS experiment_split CASCADE;
DROP TABLE IF EXISTS ensemble_components CASCADE;
DROP TABLE IF EXISTS model_artifacts CASCADE;
DROP TABLE IF EXISTS ensemble_experiments CASCADE;
DROP TABLE IF EXISTS data_observations CASCADE;

CREATE TABLE data_observations (
    observation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_name TEXT NOT NULL,
    external_key TEXT NOT NULL,
    target_label SMALLINT NOT NULL CHECK (target_label IN (0, 1)),
    feature_payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (dataset_name, external_key)
);

CREATE TABLE ensemble_experiments (
    experiment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_name TEXT NOT NULL UNIQUE,
    ensemble_type TEXT NOT NULL CHECK (
        ensemble_type IN (
            'voting',
            'bagging',
            'boosting',
            'stacking',
            'blending'
        )
    ),
    task_type TEXT NOT NULL CHECK (
        task_type IN ('classification', 'regression')
    ),
    validation_strategy TEXT NOT NULL CHECK (
        validation_strategy IN (
            'holdout',
            'k_fold',
            'stratified_k_fold',
            'out_of_fold'
        )
    ),
    random_seed INTEGER NOT NULL,
    description TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE model_artifacts (
    model_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_name TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    model_version TEXT NOT NULL,
    training_dataset TEXT NOT NULL,
    hyperparameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (model_name, model_version)
);

CREATE TABLE ensemble_components (
    experiment_id BIGINT NOT NULL REFERENCES ensemble_experiments(experiment_id)
        ON DELETE CASCADE,
    model_id BIGINT NOT NULL REFERENCES model_artifacts(model_id),
    component_role TEXT NOT NULL CHECK (
        component_role IN ('base', 'meta', 'weak_learner')
    ),
    aggregation_weight NUMERIC(12, 8) NOT NULL DEFAULT 1.0
        CHECK (aggregation_weight >= 0),
    component_order INTEGER NOT NULL CHECK (component_order >= 0),
    PRIMARY KEY (experiment_id, model_id),
    UNIQUE (experiment_id, component_order)
);

CREATE TABLE experiment_split (
    experiment_id BIGINT NOT NULL REFERENCES ensemble_experiments(experiment_id)
        ON DELETE CASCADE,
    observation_id BIGINT NOT NULL REFERENCES data_observations(observation_id)
        ON DELETE CASCADE,
    split_name TEXT NOT NULL CHECK (
        split_name IN ('train', 'validation', 'test')
    ),
    fold_number INTEGER CHECK (fold_number IS NULL OR fold_number >= 0),
    PRIMARY KEY (experiment_id, observation_id)
);

CREATE TABLE ensemble_predictions (
    prediction_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id BIGINT NOT NULL REFERENCES ensemble_experiments(experiment_id)
        ON DELETE CASCADE,
    observation_id BIGINT NOT NULL REFERENCES data_observations(observation_id),
    model_id BIGINT NOT NULL REFERENCES model_artifacts(model_id),
    prediction_stage TEXT NOT NULL CHECK (
        prediction_stage IN ('base', 'meta', 'ensemble')
    ),
    predicted_probability NUMERIC(12, 10) NOT NULL
        CHECK (predicted_probability BETWEEN 0 AND 1),
    predicted_label SMALLINT NOT NULL CHECK (predicted_label IN (0, 1)),
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (experiment_id, observation_id, model_id, prediction_stage)
);

CREATE TABLE evaluation_metrics (
    metric_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id BIGINT NOT NULL REFERENCES ensemble_experiments(experiment_id)
        ON DELETE CASCADE,
    dataset_partition TEXT NOT NULL CHECK (
        dataset_partition IN ('validation', 'test')
    ),
    metric_name TEXT NOT NULL CHECK (
        metric_name IN (
            'accuracy',
            'precision',
            'recall',
            'f1',
            'roc_auc',
            'log_loss',
            'mean_absolute_error',
            'root_mean_squared_error'
        )
    ),
    metric_value NUMERIC(16, 8) NOT NULL,
    sample_count INTEGER NOT NULL CHECK (sample_count > 0),
    measured_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (experiment_id, dataset_partition, metric_name)
);

CREATE INDEX idx_observations_dataset
    ON data_observations (dataset_name, observation_id);

CREATE INDEX idx_predictions_experiment_stage
    ON ensemble_predictions (experiment_id, prediction_stage, observation_id);

CREATE INDEX idx_splits_experiment_partition
    ON experiment_split (experiment_id, split_name, observation_id);

CREATE INDEX idx_metrics_experiment_partition
    ON evaluation_metrics (experiment_id, dataset_partition, metric_name);

-- Each ensemble type has a different aggregation mechanism.
INSERT INTO ensemble_experiments (
    experiment_name,
    ensemble_type,
    task_type,
    validation_strategy,
    random_seed,
    description
)
VALUES
(
    'payment-risk-voting',
    'voting',
    'classification',
    'holdout',
    101,
    'Combines heterogeneous base-model predictions using weighted voting.'
),
(
    'payment-risk-bagging',
    'bagging',
    'classification',
    'stratified_k_fold',
    102,
    'Fits independent base learners on bootstrap samples and aggregates them.'
),
(
    'payment-risk-boosting',
    'boosting',
    'classification',
    'stratified_k_fold',
    103,
    'Sequential weak learners emphasize examples misclassified in earlier rounds.'
),
(
    'payment-risk-stacking',
    'stacking',
    'classification',
    'out_of_fold',
    104,
    'Trains a meta-model on base-model out-of-fold predictions.'
),
(
    'payment-risk-blending',
    'blending',
    'classification',
    'holdout',
    105,
    'Uses a dedicated validation partition to fit a prediction combiner.'
);

INSERT INTO model_artifacts (
    model_name,
    algorithm,
    model_version,
    training_dataset,
    hyperparameters
)
VALUES
(
    'risk-logistic',
    'logistic_regression',
    '1.0',
    'payments-v3',
    '{"regularization": 0.01, "learning_rate": 0.1}'::jsonb
),
(
    'risk-tree',
    'decision_tree',
    '1.2',
    'payments-v3',
    '{"max_depth": 4, "min_samples_leaf": 10}'::jsonb
),
(
    'risk-forest',
    'random_forest',
    '2.0',
    'payments-v3',
    '{"n_estimators": 200, "max_features": "sqrt"}'::jsonb
),
(
    'risk-boosted-trees',
    'gradient_boosted_trees',
    '1.4',
    'payments-v3',
    '{"learning_rate": 0.05, "max_depth": 3}'::jsonb
),
(
    'risk-meta',
    'logistic_regression',
    '1.0',
    'payments-oof-v3',
    '{"regularization": 0.1, "input": "out_of_fold_predictions"}'::jsonb
),
(
    'risk-blender',
    'logistic_regression',
    '1.0',
    'payments-validation-v3',
    '{"input": "held_out_base_predictions"}'::jsonb
);

-- Component membership is normalized rather than embedded in a JSON array.
INSERT INTO ensemble_components (
    experiment_id,
    model_id,
    component_role,
    aggregation_weight,
    component_order
)
SELECT e.experiment_id, m.model_id, x.component_role,
       x.aggregation_weight, x.component_order
FROM (
    VALUES
        ('payment-risk-voting', 'risk-logistic', 'base', 1.2::numeric, 0),
        ('payment-risk-voting', 'risk-tree', 'base', 0.8::numeric, 1),
        ('payment-risk-bagging', 'risk-tree', 'base', 1.0::numeric, 0),
        ('payment-risk-bagging', 'risk-forest', 'base', 1.0::numeric, 1),
        ('payment-risk-boosting', 'risk-boosted-trees', 'weak_learner', 1.0::numeric, 0),
        ('payment-risk-stacking', 'risk-logistic', 'base', 1.0::numeric, 0),
        ('payment-risk-stacking', 'risk-forest', 'base', 1.0::numeric, 1),
        ('payment-risk-stacking', 'risk-meta', 'meta', 1.0::numeric, 2),
        ('payment-risk-blending', 'risk-logistic', 'base', 1.0::numeric, 0),
        ('payment-risk-blending', 'risk-forest', 'base', 1.0::numeric, 1),
        ('payment-risk-blending', 'risk-blender', 'meta', 1.0::numeric, 2)
) AS x(experiment_name, model_name, component_role,
       aggregation_weight, component_order)
JOIN ensemble_experiments e ON e.experiment_name = x.experiment_name
JOIN model_artifacts m ON m.model_name = x.model_name;

INSERT INTO data_observations (
    dataset_name,
    external_key,
    target_label,
    feature_payload
)
VALUES
    ('payments-v3', 'PAY-001', 0, '{"amount": 35.00, "account_age_days": 900, "velocity": 1}'::jsonb),
    ('payments-v3', 'PAY-002', 1, '{"amount": 2700.00, "account_age_days": 3, "velocity": 14}'::jsonb),
    ('payments-v3', 'PAY-003', 0, '{"amount": 120.00, "account_age_days": 500, "velocity": 2}'::jsonb),
    ('payments-v3', 'PAY-004', 1, '{"amount": 1800.00, "account_age_days": 8, "velocity": 11}'::jsonb),
    ('payments-v3', 'PAY-005', 0, '{"amount": 65.00, "account_age_days": 1200, "velocity": 0}'::jsonb),
    ('payments-v3', 'PAY-006', 1, '{"amount": 2300.00, "account_age_days": 12, "velocity": 9}'::jsonb);

-- Demonstration partitions are disjoint by experiment and observation.
-- In production, assign partitions before any model or preprocessing fit.
INSERT INTO experiment_split (
    experiment_id,
    observation_id,
    split_name,
    fold_number
)
SELECT e.experiment_id, o.observation_id,
       CASE
           WHEN o.external_key IN ('PAY-001', 'PAY-002', 'PAY-003')
               THEN 'train'
           WHEN o.external_key = 'PAY-004'
               THEN 'validation'
           ELSE 'test'
       END,
       CASE
           WHEN o.external_key IN ('PAY-001', 'PAY-002') THEN 0
           WHEN o.external_key = 'PAY-003' THEN 1
           ELSE NULL
       END
FROM ensemble_experiments e
CROSS JOIN data_observations o
WHERE o.dataset_name = 'payments-v3';

-- Store representative predictions for one experiment.
-- These rows illustrate provenance and database constraints, not a claim
-- that the numerical predictions were produced by a fitted production model.
INSERT INTO ensemble_predictions (
    experiment_id,
    observation_id,
    model_id,
    prediction_stage,
    predicted_probability,
    predicted_label
)
SELECT e.experiment_id, o.observation_id, m.model_id,
       'base',
       v.probability,
       CASE WHEN v.probability >= 0.5 THEN 1 ELSE 0 END
FROM (
    VALUES
        ('PAY-004', 'risk-logistic', 0.82::numeric),
        ('PAY-004', 'risk-forest', 0.76::numeric),
        ('PAY-005', 'risk-logistic', 0.08::numeric),
        ('PAY-005', 'risk-forest', 0.14::numeric),
        ('PAY-006', 'risk-logistic', 0.91::numeric),
        ('PAY-006', 'risk-forest', 0.84::numeric)
) AS v(external_key, model_name, probability)
JOIN data_observations o
    ON o.external_key = v.external_key
JOIN ensemble_experiments e
    ON e.experiment_name = 'payment-risk-stacking'
JOIN model_artifacts m
    ON m.model_name = v.model_name;

INSERT INTO evaluation_metrics (
    experiment_id,
    dataset_partition,
    metric_name,
    metric_value,
    sample_count
)
SELECT e.experiment_id, v.partition, v.metric, v.value, v.sample_count
FROM (
    VALUES
        ('payment-risk-voting', 'validation', 'accuracy', 0.88::numeric, 120),
        ('payment-risk-voting', 'test', 'accuracy', 0.86::numeric, 140),
        ('payment-risk-bagging', 'validation', 'accuracy', 0.90::numeric, 120),
        ('payment-risk-bagging', 'test', 'accuracy', 0.89::numeric, 140),
        ('payment-risk-boosting', 'validation', 'accuracy', 0.91::numeric, 120),
        ('payment-risk-boosting', 'test', 'accuracy', 0.88::numeric, 140),
        ('payment-risk-stacking', 'validation', 'log_loss', 0.31::numeric, 120),
        ('payment-risk-stacking', 'test', 'log_loss', 0.35::numeric, 140),
        ('payment-risk-blending', 'validation', 'log_loss', 0.29::numeric, 120),
        ('payment-risk-blending', 'test', 'log_loss', 0.33::numeric, 140)
) AS v(experiment_name, partition, metric, value, sample_count)
JOIN ensemble_experiments e ON e.experiment_name = v.experiment_name;

CREATE VIEW ensemble_metric_comparison AS
SELECT
    e.experiment_name,
    e.ensemble_type,
    m.dataset_partition,
    m.metric_name,
    m.metric_value,
    m.sample_count,
    m.measured_at
FROM evaluation_metrics m
JOIN ensemble_experiments e USING (experiment_id);

CREATE VIEW ensemble_prediction_summary AS
SELECT
    e.experiment_name,
    p.prediction_stage,
    COUNT(*) AS prediction_count,
    ROUND(AVG(p.predicted_probability), 4) AS mean_predicted_probability,
    SUM(CASE WHEN p.predicted_label = 1 THEN 1 ELSE 0 END)
        AS positive_predictions
FROM ensemble_predictions p
JOIN ensemble_experiments e USING (experiment_id)
GROUP BY e.experiment_name, p.prediction_stage;

-- Compare measured performance on the held-out test partition.
SELECT
    experiment_name,
    ensemble_type,
    metric_name,
    metric_value,
    sample_count
FROM ensemble_metric_comparison
WHERE dataset_partition = 'test'
ORDER BY metric_name, metric_value;

-- Inspect component composition and assigned aggregation weights.
SELECT
    e.experiment_name,
    e.ensemble_type,
    m.model_name,
    m.algorithm,
    c.component_role,
    c.aggregation_weight,
    c.component_order
FROM ensemble_components c
JOIN ensemble_experiments e USING (experiment_id)
JOIN model_artifacts m USING (model_id)
ORDER BY e.experiment_name, c.component_order;

-- Examine prediction distributions without confusing predictions with outcomes.
SELECT *
FROM ensemble_prediction_summary
ORDER BY experiment_name, prediction_stage;

-- Detect observations incorrectly assigned to more than one partition.
-- The primary key on (experiment_id, observation_id) prevents this state.
SELECT experiment_id, observation_id, COUNT(*) AS partition_count
FROM experiment_split
GROUP BY experiment_id, observation_id
HAVING COUNT(*) > 1;

-- Find test observations that have recorded ensemble predictions.
SELECT
    e.experiment_name,
    o.external_key,
    o.target_label,
    p.predicted_probability,
    p.predicted_label,
    p.prediction_stage
FROM experiment_split s
JOIN ensemble_experiments e USING (experiment_id)
JOIN data_observations o USING (observation_id)
LEFT JOIN ensemble_predictions p
    ON p.experiment_id = s.experiment_id
   AND p.observation_id = s.observation_id
WHERE s.split_name = 'test'
ORDER BY e.experiment_name, o.external_key;

-- Example transaction: insert a valid model evaluation atomically.
BEGIN;

INSERT INTO evaluation_metrics (
    experiment_id,
    dataset_partition,
    metric_name,
    metric_value,
    sample_count
)
SELECT experiment_id, 'test', 'precision', 0.87, 140
FROM ensemble_experiments
WHERE experiment_name = 'payment-risk-bagging'
ON CONFLICT (experiment_id, dataset_partition, metric_name)
DO UPDATE SET
    metric_value = EXCLUDED.metric_value,
    sample_count = EXCLUDED.sample_count,
    measured_at = CURRENT_TIMESTAMP;

COMMIT;

-- Database validation examples:
-- The following constraints reject invalid experiment types, negative weights,
-- non-binary labels, invalid probabilities, and duplicate split assignments.
-- Run each deliberately invalid statement separately when testing failures.
--
-- INSERT INTO ensemble_experiments
--     (experiment_name, ensemble_type, task_type, validation_strategy,
--      random_seed, description)
-- VALUES
--     ('invalid-ensemble', 'averaging-only', 'classification',
--      'holdout', 1, 'Rejected: unsupported ensemble type.');
--
-- INSERT INTO ensemble_predictions
--     (experiment_id, observation_id, model_id, prediction_stage,
--      predicted_probability, predicted_label)
-- VALUES
--     (1, 1, 1, 'base', 1.7, 1);
--
-- INSERT INTO ensemble_components
--     (experiment_id, model_id, component_role, aggregation_weight, component_order)
-- VALUES
--     (1, 1, 'base', -0.5, 99);

COMMIT;
