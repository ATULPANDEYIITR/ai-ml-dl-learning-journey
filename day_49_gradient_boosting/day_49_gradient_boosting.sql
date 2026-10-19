DROP SCHEMA IF EXISTS gradient_boosting_lab CASCADE;
CREATE SCHEMA gradient_boosting_lab;
SET search_path TO gradient_boosting_lab;

-- PostgreSQL model for a small gradient-boosting training and evaluation
-- workflow. The relational representation keeps observations, weak learners,
-- residuals, learning-rate contributions, and predictions auditable.

CREATE TABLE training_observations (
    observation_id BIGSERIAL PRIMARY KEY,
    sample_name TEXT NOT NULL UNIQUE,
    payload_kb NUMERIC(12, 2) NOT NULL CHECK (payload_kb >= 0),
    concurrent_requests INTEGER NOT NULL CHECK (concurrent_requests >= 0),
    database_queries INTEGER NOT NULL CHECK (database_queries >= 0),
    cache_hit_percent NUMERIC(5, 2) NOT NULL
        CHECK (cache_hit_percent BETWEEN 0 AND 100),
    target_latency_ms NUMERIC(12, 4) NOT NULL CHECK (target_latency_ms >= 0)
);

CREATE TABLE boosting_runs (
    run_id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    learning_rate NUMERIC(10, 6) NOT NULL
        CHECK (learning_rate > 0 AND learning_rate <= 1),
    max_tree_depth INTEGER NOT NULL CHECK (max_tree_depth >= 0),
    estimator_count INTEGER NOT NULL CHECK (estimator_count > 0),
    initial_prediction NUMERIC(14, 6),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE weak_learners (
    tree_id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES boosting_runs(run_id)
        ON DELETE CASCADE,
    tree_number INTEGER NOT NULL CHECK (tree_number > 0),
    feature_name TEXT NOT NULL,
    threshold NUMERIC(14, 6) NOT NULL,
    leaf_left_prediction NUMERIC(14, 6) NOT NULL,
    leaf_right_prediction NUMERIC(14, 6) NOT NULL,
    residual_sse_reduction NUMERIC(18, 6) NOT NULL
        CHECK (residual_sse_reduction >= 0),
    UNIQUE (run_id, tree_number)
);

CREATE TABLE residual_snapshots (
    run_id BIGINT NOT NULL REFERENCES boosting_runs(run_id)
        ON DELETE CASCADE,
    tree_number INTEGER NOT NULL CHECK (tree_number > 0),
    observation_id BIGINT NOT NULL
        REFERENCES training_observations(observation_id)
        ON DELETE CASCADE,
    current_prediction NUMERIC(14, 6) NOT NULL,
    residual NUMERIC(14, 6) NOT NULL,
    scaled_correction NUMERIC(14, 6) NOT NULL,
    PRIMARY KEY (run_id, tree_number, observation_id)
);

CREATE TABLE predictions (
    run_id BIGINT NOT NULL REFERENCES boosting_runs(run_id)
        ON DELETE CASCADE,
    observation_id BIGINT NOT NULL
        REFERENCES training_observations(observation_id)
        ON DELETE CASCADE,
    prediction NUMERIC(14, 6) NOT NULL,
    squared_error NUMERIC(18, 6) NOT NULL
        CHECK (squared_error >= 0),
    PRIMARY KEY (run_id, observation_id)
);

CREATE INDEX idx_weak_learners_run_tree
    ON weak_learners (run_id, tree_number);

CREATE INDEX idx_residual_snapshots_observation
    ON residual_snapshots (observation_id, run_id);

CREATE INDEX idx_predictions_run_error
    ON predictions (run_id, squared_error DESC);

INSERT INTO training_observations (
    sample_name,
    payload_kb,
    concurrent_requests,
    database_queries,
    cache_hit_percent,
    target_latency_ms
)
VALUES
    ('baseline-small', 20, 5, 2, 92, 42),
    ('normal-small', 40, 10, 3, 88, 58),
    ('normal-medium', 60, 15, 4, 84, 73),
    ('high-medium', 80, 20, 6, 78, 98),
    ('high-large', 100, 25, 7, 70, 124),
    ('large-1', 120, 30, 9, 64, 151),
    ('large-2', 140, 35, 11, 58, 180),
    ('large-3', 160, 40, 12, 50, 214),
    ('heavy-1', 180, 45, 15, 45, 249),
    ('heavy-2', 200, 50, 17, 38, 290),
    ('cache-efficient', 30, 8, 2, 94, 49),
    ('moderate-load', 70, 18, 5, 82, 88),
    ('database-heavy', 110, 28, 8, 68, 137),
    ('service-heavy', 150, 38, 12, 55, 192),
    ('extreme-load', 190, 48, 16, 42, 264),
    ('small-burst', 50, 12, 3, 90, 65),
    ('medium-burst', 90, 22, 6, 75, 112),
    ('large-burst', 130, 32, 10, 60, 165);

INSERT INTO boosting_runs (
    model_name,
    learning_rate,
    max_tree_depth,
    estimator_count,
    initial_prediction
)
SELECT
    'api-latency-gbdt',
    0.04,
    2,
    6,
    AVG(target_latency_ms)
FROM training_observations;

-- The initial model is the mean target. This is the zero-tree baseline
-- against which residual fitting begins.
WITH baseline AS (
    SELECT AVG(target_latency_ms) AS initial_prediction
    FROM training_observations
)
SELECT
    sample_name,
    target_latency_ms,
    initial_prediction,
    target_latency_ms - initial_prediction AS residual
FROM training_observations
CROSS JOIN baseline
ORDER BY observation_id;

-- A depth-one weak learner can partition observations using a single
-- threshold. This query evaluates candidate thresholds by residual SSE.
WITH baseline AS (
    SELECT AVG(target_latency_ms) AS prediction
    FROM training_observations
),
residuals AS (
    SELECT
        o.*,
        o.target_latency_ms - b.prediction AS residual
    FROM training_observations o
    CROSS JOIN baseline b
),
candidate_thresholds AS (
    SELECT DISTINCT
        payload_kb AS threshold
    FROM training_observations
),
split_quality AS (
    SELECT
        c.threshold,
        SUM(
            CASE
                WHEN r.payload_kb <= c.threshold THEN
                    POWER(
                        r.residual
                        - AVG(r.residual) OVER (
                            PARTITION BY
                                CASE
                                    WHEN r.payload_kb <= c.threshold
                                    THEN 0
                                    ELSE 1
                                END
                        ),
                        2
                    )
                ELSE 0
            END
        ) AS within_group_sse
    FROM candidate_thresholds c
    CROSS JOIN residuals r
    GROUP BY c.threshold
)
SELECT
    threshold,
    within_group_sse
FROM split_quality
ORDER BY within_group_sse
LIMIT 5;

-- Store a representative sequence of weak learners. Each tree is deliberately
-- shallow; its job is to make a modest correction rather than model the target
-- independently.
INSERT INTO weak_learners (
    run_id,
    tree_number,
    feature_name,
    threshold,
    leaf_left_prediction,
    leaf_right_prediction,
    residual_sse_reduction
)
SELECT
    (SELECT run_id FROM boosting_runs WHERE model_name = 'api-latency-gbdt'),
    v.tree_number,
    v.feature_name,
    v.threshold,
    v.left_prediction,
    v.right_prediction,
    v.sse_reduction
FROM (
    VALUES
        (1, 'payload_kb', 105.0, -35.0, 35.0, 42000.0),
        (2, 'cache_hit_percent', 72.0, 24.0, -22.0, 18000.0),
        (3, 'concurrent_requests', 27.0, -18.0, 20.0, 9500.0),
        (4, 'database_queries', 8.0, -11.0, 13.0, 4700.0),
        (5, 'payload_kb', 155.0, -8.0, 10.0, 2500.0),
        (6, 'cache_hit_percent', 55.0, 7.0, -6.0, 1200.0)
) AS v(
    tree_number,
    feature_name,
    threshold,
    left_prediction,
    right_prediction,
    sse_reduction
);

-- Demonstrate the mathematical update:
-- F_t(x) = F_(t-1)(x) + learning_rate * h_t(x)
WITH run AS (
    SELECT *
    FROM boosting_runs
    WHERE model_name = 'api-latency-gbdt'
),
initial AS (
    SELECT
        run_id,
        initial_prediction,
        learning_rate
    FROM run
),
updates AS (
    SELECT
        t.observation_id,
        i.run_id,
        i.initial_prediction
        + SUM(
            i.learning_rate
            * CASE
                WHEN
                    CASE w.feature_name
                        WHEN 'payload_kb'
                            THEN t.payload_kb
                        WHEN 'cache_hit_percent'
                            THEN t.cache_hit_percent
                        WHEN 'concurrent_requests'
                            THEN t.concurrent_requests
                        WHEN 'database_queries'
                            THEN t.database_queries
                    END
                    <= w.threshold
                THEN w.leaf_left_prediction
                ELSE w.leaf_right_prediction
              END
        ) AS prediction
    FROM training_observations t
    CROSS JOIN initial i
    JOIN weak_learners w
      ON w.run_id = i.run_id
    GROUP BY
        t.observation_id,
        i.run_id,
        i.initial_prediction
)
SELECT
    t.sample_name,
    t.target_latency_ms,
    ROUND(u.prediction, 3) AS boosted_prediction,
    ROUND(
        POWER(t.target_latency_ms - u.prediction, 2),
        3
    ) AS squared_error
FROM updates u
JOIN training_observations t
  ON t.observation_id = u.observation_id
ORDER BY squared_error DESC;

-- Learning rate is stored as model configuration rather than embedded in
-- individual observations. This makes the contribution of every weak learner
-- auditable and allows configurations to be compared.
SELECT
    model_name,
    learning_rate,
    max_tree_depth,
    estimator_count,
    initial_prediction
FROM boosting_runs
ORDER BY created_at;

-- Feature importance here is based on the amount of residual SSE reduction
-- attributed to splits on each feature. It should not be interpreted as
-- causal importance.
SELECT
    feature_name,
    SUM(residual_sse_reduction) AS total_residual_sse_reduction,
    ROUND(
        SUM(residual_sse_reduction)
        / SUM(SUM(residual_sse_reduction))
          OVER (),
        4
    ) AS normalized_importance
FROM weak_learners
WHERE run_id = (
    SELECT run_id
    FROM boosting_runs
    WHERE model_name = 'api-latency-gbdt'
)
GROUP BY feature_name
ORDER BY normalized_importance DESC;

-- Training-loss style evaluation using the stored final predictions.
-- The aggregate makes model error visible without changing training data.
WITH final_predictions AS (
    SELECT
        p.run_id,
        p.observation_id,
        p.prediction,
        o.target_latency_ms
    FROM predictions p
    JOIN training_observations o
      ON o.observation_id = p.observation_id
)
SELECT
    run_id,
    COUNT(*) AS observations,
    AVG(POWER(target_latency_ms - prediction, 2))
        AS mean_squared_error,
    SQRT(
        AVG(POWER(target_latency_ms - prediction, 2))
    ) AS root_mean_squared_error
FROM final_predictions
GROUP BY run_id;

-- PostgreSQL transaction demonstration. The model configuration and its
-- weak learners should be committed together so an incomplete model is not
-- published as a valid training run.
BEGIN;

INSERT INTO boosting_runs (
    model_name,
    learning_rate,
    max_tree_depth,
    estimator_count,
    initial_prediction
)
SELECT
    'transactional-validation-run',
    0.02,
    1,
    20,
    AVG(target_latency_ms)
FROM training_observations;

COMMIT;

-- Integrity examples:
-- The following statements would be rejected by CHECK constraints:
-- INSERT INTO training_observations
--     (sample_name, payload_kb, concurrent_requests,
--      database_queries, cache_hit_percent, target_latency_ms)
-- VALUES ('invalid-cache', 50, 10, 3, 125, 70);
--
-- INSERT INTO boosting_runs
--     (model_name, learning_rate, max_tree_depth, estimator_count)
-- VALUES ('invalid-rate', 0, 2, 50);

-- Query useful for monitoring whether a weak learner is contributing
-- meaningful residual-error reduction.
SELECT
    tree_number,
    feature_name,
    residual_sse_reduction,
    CASE
        WHEN residual_sse_reduction = 0 THEN 'no_split_gain'
        WHEN residual_sse_reduction < 2000 THEN 'small_correction'
        ELSE 'material_correction'
    END AS contribution_class
FROM weak_learners
WHERE run_id = (
    SELECT run_id
    FROM boosting_runs
    WHERE model_name = 'api-latency-gbdt'
)
ORDER BY tree_number;
