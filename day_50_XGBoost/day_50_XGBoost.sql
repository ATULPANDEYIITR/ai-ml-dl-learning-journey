DROP SCHEMA IF EXISTS xgboost_lab CASCADE;
CREATE SCHEMA xgboost_lab;
SET search_path TO xgboost_lab;

CREATE TABLE model_run (
    model_run_id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    objective TEXT NOT NULL CHECK (objective IN ('binary:logistic', 'reg:squarederror')),
    learning_rate NUMERIC(8,6) NOT NULL CHECK (learning_rate > 0 AND learning_rate <= 1),
    n_estimators INTEGER NOT NULL CHECK (n_estimators > 0),
    max_depth INTEGER NOT NULL CHECK (max_depth > 0),
    min_child_weight NUMERIC(14,6) NOT NULL CHECK (min_child_weight > 0),
    reg_alpha NUMERIC(14,6) NOT NULL DEFAULT 0 CHECK (reg_alpha >= 0),
    reg_lambda NUMERIC(14,6) NOT NULL DEFAULT 1 CHECK (reg_lambda >= 0),
    gamma NUMERIC(14,6) NOT NULL DEFAULT 0 CHECK (gamma >= 0),
    subsample NUMERIC(8,6) NOT NULL DEFAULT 1 CHECK (subsample > 0 AND subsample <= 1),
    colsample_bytree NUMERIC(8,6) NOT NULL DEFAULT 1 CHECK (colsample_bytree > 0 AND colsample_bytree <= 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE training_record (
    record_id BIGSERIAL PRIMARY KEY,
    model_run_id BIGINT NOT NULL REFERENCES model_run(model_run_id) ON DELETE CASCADE,
    changed_lines INTEGER NOT NULL CHECK (changed_lines >= 0),
    test_failure_rate NUMERIC(8,6) NOT NULL CHECK (test_failure_rate BETWEEN 0 AND 1),
    historical_risk NUMERIC(8,6) NOT NULL CHECK (historical_risk BETWEEN 0 AND 1),
    review_coverage NUMERIC(8,6) NOT NULL CHECK (review_coverage BETWEEN 0 AND 1),
    target SMALLINT NOT NULL CHECK (target IN (0, 1))
);

CREATE TABLE boosting_tree (
    tree_id BIGSERIAL PRIMARY KEY,
    model_run_id BIGINT NOT NULL REFERENCES model_run(model_run_id) ON DELETE CASCADE,
    iteration INTEGER NOT NULL CHECK (iteration >= 0),
    learning_rate NUMERIC(8,6) NOT NULL CHECK (learning_rate > 0),
    UNIQUE (model_run_id, iteration)
);

CREATE TABLE tree_node (
    node_id BIGSERIAL PRIMARY KEY,
    tree_id BIGINT NOT NULL REFERENCES boosting_tree(tree_id) ON DELETE CASCADE,
    node_position INTEGER NOT NULL CHECK (node_position >= 0),
    parent_node_id BIGINT REFERENCES tree_node(node_id) ON DELETE CASCADE,
    feature_name TEXT,
    split_threshold NUMERIC(20,8),
    leaf_weight NUMERIC(20,10),
    left_node_id BIGINT REFERENCES tree_node(node_id),
    right_node_id BIGINT REFERENCES tree_node(node_id),
    split_gain NUMERIC(20,10),
    CHECK (
        (feature_name IS NULL AND split_threshold IS NULL AND leaf_weight IS NOT NULL)
        OR
        (feature_name IS NOT NULL AND split_threshold IS NOT NULL AND leaf_weight IS NULL)
    ),
    UNIQUE (tree_id, node_position)
);

CREATE TABLE prediction (
    prediction_id BIGSERIAL PRIMARY KEY,
    model_run_id BIGINT NOT NULL REFERENCES model_run(model_run_id) ON DELETE CASCADE,
    record_id BIGINT REFERENCES training_record(record_id) ON DELETE SET NULL,
    raw_margin NUMERIC(30,15) NOT NULL,
    probability NUMERIC(30,15) NOT NULL CHECK (probability BETWEEN 0 AND 1),
    predicted_class SMALLINT NOT NULL CHECK (predicted_class IN (0, 1)),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_training_record_model_target
    ON training_record(model_run_id, target);

CREATE INDEX idx_tree_iteration
    ON boosting_tree(model_run_id, iteration);

CREATE INDEX idx_prediction_model_class
    ON prediction(model_run_id, predicted_class);

INSERT INTO model_run (
    model_name,
    objective,
    learning_rate,
    n_estimators,
    max_depth,
    min_child_weight,
    reg_alpha,
    reg_lambda,
    gamma,
    subsample,
    colsample_bytree
)
VALUES (
    'pull_request_risk_xgb',
    'binary:logistic',
    0.10,
    100,
    3,
    1.0,
    0.05,
    2.0,
    0.10,
    0.90,
    0.80
);

INSERT INTO training_record (
    model_run_id,
    changed_lines,
    test_failure_rate,
    historical_risk,
    review_coverage,
    target
)
SELECT
    1,
    changed_lines,
    test_failure_rate,
    historical_risk,
    review_coverage,
    target
FROM (
    VALUES
        (10, 0.00::numeric, 0.10::numeric, 0.90::numeric, 0),
        (20, 0.00::numeric, 0.20::numeric, 0.80::numeric, 0),
        (25, 0.02::numeric, 0.15::numeric, 0.90::numeric, 0),
        (40, 0.10::numeric, 0.40::numeric, 0.70::numeric, 1),
        (55, 0.15::numeric, 0.55::numeric, 0.60::numeric, 1),
        (70, 0.20::numeric, 0.65::numeric, 0.55::numeric, 1),
        (80, 0.25::numeric, 0.80::numeric, 0.40::numeric, 1),
        (95, 0.35::numeric, 0.90::numeric, 0.20::numeric, 1),
        (15, 0.00::numeric, 0.05::numeric, 0.95::numeric, 0),
        (35, 0.04::numeric, 0.35::numeric, 0.75::numeric, 0),
        (60, 0.10::numeric, 0.50::numeric, 0.65::numeric, 1),
        (90, 0.30::numeric, 0.85::numeric, 0.30::numeric, 1)
) AS v(changed_lines, test_failure_rate, historical_risk, review_coverage, target);

INSERT INTO boosting_tree (
    model_run_id,
    iteration,
    learning_rate
)
SELECT 1, g, 0.10
FROM generate_series(0, 9) AS g;

INSERT INTO tree_node (
    tree_id,
    node_position,
    feature_name,
    split_threshold,
    leaf_weight,
    split_gain
)
VALUES
    (1, 0, 'test_failure_rate', 0.075, NULL, 1.42),
    (1, 1, NULL, NULL, -0.31, NULL),
    (1, 2, NULL, NULL, 0.44, NULL),
    (2, 0, 'historical_risk', 0.45, NULL, 0.91),
    (2, 1, NULL, NULL, -0.20, NULL),
    (2, 2, NULL, NULL, 0.37, NULL);

CREATE OR REPLACE FUNCTION logistic_probability(margin NUMERIC)
RETURNS NUMERIC
LANGUAGE SQL
IMMUTABLE
STRICT
AS $$
    SELECT CASE
        WHEN margin >= 0
            THEN 1.0 / (1.0 + exp(-margin))
        ELSE exp(margin) / (1.0 + exp(margin))
    END;
$$;

CREATE OR REPLACE FUNCTION xgboost_leaf_weight(
    gradient_sum NUMERIC,
    hessian_sum NUMERIC,
    reg_lambda NUMERIC,
    reg_alpha NUMERIC
)
RETURNS NUMERIC
LANGUAGE plpgsql
IMMUTABLE
STRICT
AS $$
DECLARE
    adjusted_gradient NUMERIC;
BEGIN
    IF hessian_sum <= 0 OR reg_lambda < 0 OR reg_alpha < 0 THEN
        RAISE EXCEPTION 'Invalid second-order objective parameters';
    END IF;

    IF gradient_sum > reg_alpha THEN
        adjusted_gradient := gradient_sum - reg_alpha;
    ELSIF gradient_sum < -reg_alpha THEN
        adjusted_gradient := gradient_sum + reg_alpha;
    ELSE
        adjusted_gradient := 0;
    END IF;

    RETURN -adjusted_gradient / (hessian_sum + reg_lambda);
END;
$$;

CREATE OR REPLACE FUNCTION xgboost_split_gain(
    left_gradient NUMERIC,
    left_hessian NUMERIC,
    right_gradient NUMERIC,
    right_hessian NUMERIC,
    parent_gradient NUMERIC,
    parent_hessian NUMERIC,
    reg_lambda NUMERIC,
    reg_alpha NUMERIC,
    gamma NUMERIC,
    min_child_weight NUMERIC
)
RETURNS NUMERIC
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    lg NUMERIC;
    rg NUMERIC;
    pg NUMERIC;
BEGIN
    IF left_hessian < min_child_weight
       OR right_hessian < min_child_weight THEN
        RETURN '-Infinity'::NUMERIC;
    END IF;

    lg := CASE
        WHEN left_gradient > reg_alpha THEN left_gradient - reg_alpha
        WHEN left_gradient < -reg_alpha THEN left_gradient + reg_alpha
        ELSE 0
    END;

    rg := CASE
        WHEN right_gradient > reg_alpha THEN right_gradient - reg_alpha
        WHEN right_gradient < -reg_alpha THEN right_gradient + reg_alpha
        ELSE 0
    END;

    pg := CASE
        WHEN parent_gradient > reg_alpha THEN parent_gradient - reg_alpha
        WHEN parent_gradient < -reg_alpha THEN parent_gradient + reg_alpha
        ELSE 0
    END;

    RETURN 0.5 * (
        lg * lg / (left_hessian + reg_lambda)
        +
        rg * rg / (right_hessian + reg_lambda)
        -
        pg * pg / (parent_hessian + reg_lambda)
    ) - gamma;
END;
$$;

CREATE OR REPLACE VIEW model_configuration AS
SELECT
    model_run_id,
    model_name,
    objective,
    learning_rate,
    n_estimators,
    max_depth,
    min_child_weight,
    reg_alpha,
    reg_lambda,
    gamma,
    subsample,
    colsample_bytree,
    CASE
        WHEN learning_rate <= 0.05 THEN 'slow learning'
        WHEN learning_rate <= 0.20 THEN 'moderate learning'
        ELSE 'aggressive learning'
    END AS learning_rate_profile
FROM model_run;

CREATE OR REPLACE VIEW training_class_distribution AS
SELECT
    model_run_id,
    target,
    COUNT(*) AS records,
    ROUND(
        COUNT(*) * 100.0 /
        SUM(COUNT(*)) OVER (PARTITION BY model_run_id),
        2
    ) AS percentage
FROM training_record
GROUP BY model_run_id, target;

CREATE OR REPLACE VIEW tree_complexity AS
SELECT
    bt.model_run_id,
    bt.iteration,
    COUNT(tn.node_id) AS nodes,
    COUNT(tn.node_id) FILTER (
        WHERE tn.feature_name IS NOT NULL
    ) AS split_nodes,
    COUNT(tn.node_id) FILTER (
        WHERE tn.feature_name IS NULL
    ) AS leaf_nodes,
    MAX(tn.split_gain) AS largest_split_gain
FROM boosting_tree bt
LEFT JOIN tree_node tn ON tn.tree_id = bt.tree_id
GROUP BY bt.model_run_id, bt.iteration;

BEGIN;

WITH objective_values AS (
    SELECT
        target,
        0.5::NUMERIC AS probability
    FROM training_record
    WHERE model_run_id = 1
),
second_order_terms AS (
    SELECT
        target,
        probability,
        probability - target AS gradient,
        GREATEST(probability * (1 - probability), 0.000001) AS hessian
    FROM objective_values
)
SELECT
    COUNT(*) AS observations,
    ROUND(SUM(gradient), 6) AS gradient_sum,
    ROUND(SUM(hessian), 6) AS hessian_sum,
    ROUND(
        xgboost_leaf_weight(
            SUM(gradient),
            SUM(hessian),
            2.0,
            0.05
        ),
        6
    ) AS regularized_initial_leaf_weight
FROM second_order_terms;

COMMIT;

SELECT
    xgboost_split_gain(
        2.10,
        3.20,
        -3.80,
        2.90,
        -1.70,
        6.10,
        2.0,
        0.05,
        0.10,
        1.0
    ) AS candidate_split_gain;

SELECT
    model_name,
    learning_rate,
    n_estimators,
    max_depth,
    reg_alpha,
    reg_lambda,
    gamma,
    subsample,
    colsample_bytree
FROM model_configuration
WHERE model_run_id = 1;

SELECT
    target,
    records,
    percentage
FROM training_class_distribution
WHERE model_run_id = 1
ORDER BY target;

SELECT
    iteration,
    nodes,
    split_nodes,
    leaf_nodes,
    largest_split_gain
FROM tree_complexity
WHERE model_run_id = 1
ORDER BY iteration;

INSERT INTO prediction (
    model_run_id,
    record_id,
    raw_margin,
    probability,
    predicted_class
)
SELECT
    1,
    record_id,
    CASE
        WHEN historical_risk + test_failure_rate >= 0.70
            THEN 1.15
        ELSE -1.15
    END AS raw_margin,
    logistic_probability(
        CASE
            WHEN historical_risk + test_failure_rate >= 0.70
                THEN 1.15
            ELSE -1.15
        END
    ),
    CASE
        WHEN historical_risk + test_failure_rate >= 0.70 THEN 1
        ELSE 0
    END
FROM training_record
WHERE model_run_id = 1;

SELECT
    predicted_class,
    COUNT(*) AS predictions,
    ROUND(AVG(probability), 4) AS average_probability
FROM prediction
WHERE model_run_id = 1
GROUP BY predicted_class
ORDER BY predicted_class;

SELECT
    COUNT(*) FILTER (
        WHERE predicted_class = target
    ) AS correct_predictions,
    COUNT(*) AS total_predictions,
    ROUND(
        COUNT(*) FILTER (WHERE predicted_class = target) * 100.0
        / COUNT(*),
        2
    ) AS accuracy_percent
FROM prediction p
JOIN training_record r
    ON r.record_id = p.record_id
WHERE p.model_run_id = 1;
