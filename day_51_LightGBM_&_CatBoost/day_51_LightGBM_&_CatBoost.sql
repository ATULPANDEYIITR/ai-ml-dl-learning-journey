-- PostgreSQL 14+ executable learning model for efficient boosting experiments.
--
-- The relational model records:
--   datasets and observations
--   categorical feature values
--   boosting experiments
--   histogram configuration
--   ordered target statistics
--   trees and split decisions
--   model evaluation
--
-- PostgreSQL constraints enforce data integrity while indexes support the
-- queries used for experiment analysis.

DROP SCHEMA IF EXISTS boosting_lab CASCADE;
CREATE SCHEMA boosting_lab;

SET search_path = boosting_lab;

CREATE TABLE dataset (
    dataset_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_name TEXT NOT NULL UNIQUE,
    task_type TEXT NOT NULL CHECK (task_type IN ('binary_classification', 'regression')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE customer_observation (
    observation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL REFERENCES dataset(dataset_id) ON DELETE CASCADE,
    age NUMERIC(8,2) NOT NULL CHECK (age >= 0),
    income NUMERIC(14,2) NOT NULL CHECK (income >= 0),
    tenure_months INTEGER NOT NULL CHECK (tenure_months >= 0),
    region TEXT NOT NULL CHECK (length(trim(region)) > 0),
    plan TEXT NOT NULL CHECK (length(trim(plan)) > 0),
    target SMALLINT NOT NULL CHECK (target IN (0, 1)),
    UNIQUE (dataset_id, observation_id)
);

CREATE INDEX idx_customer_dataset_target
    ON customer_observation(dataset_id, target);

CREATE INDEX idx_customer_region
    ON customer_observation(dataset_id, region);

CREATE INDEX idx_customer_plan
    ON customer_observation(dataset_id, plan);

CREATE TABLE boosting_experiment (
    experiment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL REFERENCES dataset(dataset_id),
    model_family TEXT NOT NULL CHECK (
        model_family IN ('lightgbm_style', 'catboost_style')
    ),
    learning_rate NUMERIC(8,6) NOT NULL CHECK (
        learning_rate > 0 AND learning_rate <= 1
    ),
    boosting_rounds INTEGER NOT NULL CHECK (
        boosting_rounds > 0
    ),
    max_depth INTEGER NOT NULL CHECK (
        max_depth > 0 AND max_depth <= 64
    ),
    max_bins INTEGER NOT NULL CHECK (
        max_bins >= 2 AND max_bins <= 65536
    ),
    prior_weight NUMERIC(12,4) CHECK (
        prior_weight IS NULL OR prior_weight > 0
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE categorical_statistic (
    statistic_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id BIGINT NOT NULL REFERENCES boosting_experiment(experiment_id)
        ON DELETE CASCADE,
    feature_name TEXT NOT NULL CHECK (
        feature_name IN ('region', 'plan')
    ),
    category_value TEXT NOT NULL,
    preceding_target_sum NUMERIC(18,6) NOT NULL CHECK (
        preceding_target_sum >= 0
    ),
    preceding_count INTEGER NOT NULL CHECK (
        preceding_count >= 0
    ),
    statistic_value NUMERIC(18,10) NOT NULL CHECK (
        statistic_value >= 0 AND statistic_value <= 1
    ),
    permutation_position INTEGER NOT NULL CHECK (
        permutation_position >= 0
    ),
    UNIQUE (
        experiment_id,
        feature_name,
        category_value,
        permutation_position
    )
);

CREATE INDEX idx_categorical_stat_lookup
    ON categorical_statistic(
        experiment_id,
        feature_name,
        category_value
    );

CREATE TABLE boosting_tree (
    tree_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id BIGINT NOT NULL REFERENCES boosting_experiment(experiment_id)
        ON DELETE CASCADE,
    iteration INTEGER NOT NULL CHECK (iteration >= 1),
    learning_rate NUMERIC(8,6) NOT NULL CHECK (learning_rate > 0),
    UNIQUE (experiment_id, iteration)
);

CREATE TABLE tree_split (
    split_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tree_id BIGINT NOT NULL REFERENCES boosting_tree(tree_id)
        ON DELETE CASCADE,
    node_id INTEGER NOT NULL CHECK (node_id >= 0),
    feature_name TEXT NOT NULL,
    threshold NUMERIC(18,8) NOT NULL,
    gain NUMERIC(24,12) NOT NULL CHECK (gain >= 0),
    left_row_count INTEGER NOT NULL CHECK (left_row_count >= 0),
    right_row_count INTEGER NOT NULL CHECK (right_row_count >= 0),
    UNIQUE (tree_id, node_id)
);

CREATE TABLE model_evaluation (
    evaluation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id BIGINT NOT NULL REFERENCES boosting_experiment(experiment_id)
        ON DELETE CASCADE,
    split_name TEXT NOT NULL CHECK (
        split_name IN ('train', 'validation', 'test')
    ),
    row_count INTEGER NOT NULL CHECK (row_count > 0),
    log_loss NUMERIC(18,10) NOT NULL CHECK (log_loss >= 0),
    accuracy NUMERIC(8,6) NOT NULL CHECK (
        accuracy >= 0 AND accuracy <= 1
    ),
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (experiment_id, split_name)
);

INSERT INTO dataset(dataset_name, task_type)
VALUES ('customer_churn_demo', 'binary_classification');

INSERT INTO customer_observation (
    dataset_id,
    age,
    income,
    tenure_months,
    region,
    plan,
    target
)
SELECT
    d.dataset_id,
    x.age,
    x.income,
    x.tenure_months,
    x.region,
    x.plan,
    x.target
FROM dataset AS d
CROSS JOIN (
    VALUES
        (23, 31000, 4,  'north', 'basic',      1),
        (25, 42000, 8,  'south', 'basic',      1),
        (29, 58000, 18, 'west',  'pro',        0),
        (31, 61000, 24, 'north', 'pro',        0),
        (35, 72000, 30, 'east',  'pro',        0),
        (41, 83000, 44, 'west',  'enterprise', 0),
        (48, 91000, 52, 'north', 'enterprise', 0),
        (52, 39000, 9,  'south', 'basic',      1),
        (27, 36000, 6,  'east',  'basic',      1),
        (33, 69000, 27, 'south', 'pro',        0),
        (38, 51000, 15, 'east',  'basic',      1),
        (45, 87000, 47, 'west',  'enterprise', 0),
        (22, 28000, 3,  'north', 'basic',      1),
        (57, 99000, 72, 'north', 'enterprise', 0),
        (30, 47000, 11, 'west',  'pro',        0),
        (36, 45000, 10, 'south', 'basic',      1),
        (43, 78000, 36, 'east',  'pro',        0),
        (50, 54000, 21, 'south', 'pro',        1),
        (28, 62000, 22, 'north', 'pro',        0),
        (39, 34000, 7,  'east',  'basic',      1)
) AS x(age, income, tenure_months, region, plan, target)
WHERE d.dataset_name = 'customer_churn_demo';

INSERT INTO boosting_experiment (
    dataset_id,
    model_family,
    learning_rate,
    boosting_rounds,
    max_depth,
    max_bins,
    prior_weight
)
SELECT
    dataset_id,
    'lightgbm_style',
    0.06,
    35,
    3,
    64,
    NULL
FROM dataset
WHERE dataset_name = 'customer_churn_demo';

INSERT INTO boosting_experiment (
    dataset_id,
    model_family,
    learning_rate,
    boosting_rounds,
    max_depth,
    max_bins,
    prior_weight
)
SELECT
    dataset_id,
    'catboost_style',
    0.06,
    35,
    3,
    64,
    5.0
FROM dataset
WHERE dataset_name = 'customer_churn_demo';

-- The following CTE calculates the global prior used by ordered
-- categorical statistics.
WITH training AS (
    SELECT
        observation_id,
        target,
        row_number() OVER (
            ORDER BY observation_id
        ) AS permutation_position
    FROM customer_observation
    WHERE dataset_id = (
        SELECT dataset_id
        FROM dataset
        WHERE dataset_name = 'customer_churn_demo'
    )
),
global_rate AS (
    SELECT AVG(target::NUMERIC) AS mean_target
    FROM training
),
ordered_region AS (
    SELECT
        t.observation_id,
        t.permutation_position,
        c.region,
        t.target,
        COALESCE(
            SUM(t.target) OVER (
                PARTITION BY c.region
                ORDER BY t.permutation_position
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ),
            0
        ) AS preceding_sum,
        COALESCE(
            COUNT(t.target) OVER (
                PARTITION BY c.region
                ORDER BY t.permutation_position
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ),
            0
        ) AS preceding_count
    FROM training AS t
    JOIN customer_observation AS c
      ON c.observation_id = t.observation_id
)
SELECT
    observation_id,
    region,
    permutation_position,
    preceding_sum,
    preceding_count,
    (
        preceding_sum + 5.0 * global_rate.mean_target
    ) / (
        preceding_count + 5.0
    ) AS ordered_target_statistic
FROM ordered_region
CROSS JOIN global_rate
ORDER BY permutation_position;

-- Store representative ordered statistics for the CatBoost-style experiment.
WITH experiment AS (
    SELECT experiment_id
    FROM boosting_experiment
    WHERE model_family = 'catboost_style'
),
training AS (
    SELECT
        observation_id,
        target,
        row_number() OVER (ORDER BY observation_id) AS permutation_position
    FROM customer_observation
    WHERE dataset_id = (
        SELECT dataset_id
        FROM dataset
        WHERE dataset_name = 'customer_churn_demo'
    )
),
global_rate AS (
    SELECT AVG(target::NUMERIC) AS mean_target
    FROM training
),
region_stats AS (
    SELECT
        t.permutation_position,
        c.region AS category_value,
        SUM(t.target) OVER (
            PARTITION BY c.region
            ORDER BY t.permutation_position
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS preceding_sum,
        COUNT(t.target) OVER (
            PARTITION BY c.region
            ORDER BY t.permutation_position
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS preceding_count
    FROM training AS t
    JOIN customer_observation AS c
      ON c.observation_id = t.observation_id
)
INSERT INTO categorical_statistic (
    experiment_id,
    feature_name,
    category_value,
    preceding_target_sum,
    preceding_count,
    statistic_value,
    permutation_position
)
SELECT
    experiment.experiment_id,
    'region',
    region_stats.category_value,
    COALESCE(region_stats.preceding_sum, 0),
    COALESCE(region_stats.preceding_count, 0),
    (
        COALESCE(region_stats.preceding_sum, 0)
        + 5.0 * global_rate.mean_target
    ) / (
        COALESCE(region_stats.preceding_count, 0) + 5.0
    ),
    region_stats.permutation_position
FROM region_stats
CROSS JOIN global_rate
CROSS JOIN experiment;

-- Register representative boosting trees and split evaluations.
INSERT INTO boosting_tree (
    experiment_id,
    iteration,
    learning_rate
)
SELECT
    experiment_id,
    iteration,
    learning_rate
FROM boosting_experiment
CROSS JOIN generate_series(1, 5) AS iteration;

INSERT INTO tree_split (
    tree_id,
    node_id,
    feature_name,
    threshold,
    gain,
    left_row_count,
    right_row_count
)
SELECT
    tree.tree_id,
    0,
    CASE
        WHEN tree.iteration % 3 = 0 THEN 'income'
        WHEN tree.iteration % 3 = 1 THEN 'tenure_months'
        ELSE 'age'
    END,
    CASE
        WHEN tree.iteration % 3 = 0 THEN 60000
        WHEN tree.iteration % 3 = 1 THEN 20
        ELSE 34
    END,
    2.5 / tree.iteration,
    8,
    7
FROM boosting_tree AS tree;

INSERT INTO model_evaluation (
    experiment_id,
    split_name,
    row_count,
    log_loss,
    accuracy
)
SELECT
    experiment_id,
    'train',
    15,
    CASE
        WHEN model_family = 'catboost_style' THEN 0.21
        ELSE 0.24
    END,
    CASE
        WHEN model_family = 'catboost_style' THEN 0.866667
        ELSE 0.833333
    END
FROM boosting_experiment;

-- Compare model families without treating their categorical mechanisms
-- as identical.
SELECT
    e.model_family,
    e.learning_rate,
    e.boosting_rounds,
    e.max_depth,
    e.max_bins,
    e.prior_weight,
    m.log_loss,
    m.accuracy
FROM boosting_experiment AS e
JOIN model_evaluation AS m
  ON m.experiment_id = e.experiment_id
WHERE m.split_name = 'train'
ORDER BY m.log_loss;

-- Inspect categorical target statistics and their amount of prior evidence.
SELECT
    feature_name,
    category_value,
    MAX(preceding_count) AS maximum_prior_observations,
    ROUND(AVG(statistic_value), 4) AS average_ordered_statistic
FROM categorical_statistic
GROUP BY feature_name, category_value
ORDER BY feature_name, category_value;

-- Identify high-gain histogram splits for model diagnostics.
SELECT
    e.model_family,
    bt.iteration,
    ts.feature_name,
    ts.threshold,
    ts.gain,
    ts.left_row_count,
    ts.right_row_count
FROM tree_split AS ts
JOIN boosting_tree AS bt
  ON bt.tree_id = ts.tree_id
JOIN boosting_experiment AS e
  ON e.experiment_id = bt.experiment_id
ORDER BY ts.gain DESC;

-- A transaction demonstrates atomic experiment registration.
BEGIN;

WITH new_dataset AS (
    INSERT INTO dataset(dataset_name, task_type)
    VALUES ('transactional_validation_dataset', 'binary_classification')
    RETURNING dataset_id
)
INSERT INTO boosting_experiment (
    dataset_id,
    model_family,
    learning_rate,
    boosting_rounds,
    max_depth,
    max_bins,
    prior_weight
)
SELECT
    dataset_id,
    'lightgbm_style',
    0.05,
    20,
    4,
    32,
    NULL
FROM new_dataset;

COMMIT;

-- Database-level validation for categorical statistics.
CREATE OR REPLACE FUNCTION validate_ordered_statistic()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.statistic_value < 0 OR NEW.statistic_value > 1 THEN
        RAISE EXCEPTION
            'Ordered target statistic must be between 0 and 1';
    END IF;

    IF NEW.preceding_count = 0
       AND NEW.preceding_target_sum <> 0 THEN
        RAISE EXCEPTION
            'Zero prior observations cannot have a non-zero target sum';
    END IF;

    IF NEW.preceding_target_sum > NEW.preceding_count THEN
        RAISE EXCEPTION
            'Target sum cannot exceed prior observation count';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_ordered_statistic
BEFORE INSERT OR UPDATE ON categorical_statistic
FOR EACH ROW
EXECUTE FUNCTION validate_ordered_statistic();

-- Useful feature-distribution report for detecting category sparsity.
WITH category_counts AS (
    SELECT
        region,
        COUNT(*) AS observations,
        SUM(target) AS positive_targets
    FROM customer_observation
    GROUP BY region
)
SELECT
    region,
    observations,
    positive_targets,
    ROUND(
        positive_targets::NUMERIC / NULLIF(observations, 0),
        4
    ) AS empirical_target_rate,
    CASE
        WHEN observations < 3 THEN 'sparse'
        WHEN observations < 10 THEN 'moderate'
        ELSE 'well_observed'
    END AS category_support
FROM category_counts
ORDER BY observations;
