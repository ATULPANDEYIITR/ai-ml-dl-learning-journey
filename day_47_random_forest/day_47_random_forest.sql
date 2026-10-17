-- PostgreSQL: Random Forest experiment registry and evaluation workflow.
-- The database stores training runs, bootstrap samples, tree-level
-- importance, predictions, and out-of-bag metrics.

BEGIN;

DROP VIEW IF EXISTS random_forest_feature_ranking;
DROP VIEW IF EXISTS random_forest_run_metrics;
DROP TABLE IF EXISTS oob_predictions;
DROP TABLE IF EXISTS tree_feature_importance;
DROP TABLE IF EXISTS bootstrap_membership;
DROP TABLE IF EXISTS forest_trees;
DROP TABLE IF EXISTS inspection_observations;
DROP TABLE IF EXISTS random_forest_runs;

CREATE TABLE random_forest_runs (
    run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_name TEXT NOT NULL,
    random_seed INTEGER NOT NULL,
    tree_count INTEGER NOT NULL CHECK (tree_count > 0),
    max_depth INTEGER NOT NULL CHECK (max_depth >= 0),
    feature_subset_size INTEGER NOT NULL CHECK (feature_subset_size > 0),
    training_fraction NUMERIC(5, 4) NOT NULL
        CHECK (training_fraction > 0 AND training_fraction < 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'configured'
        CHECK (status IN ('configured', 'training', 'completed', 'failed'))
);

CREATE TABLE inspection_observations (
    observation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES random_forest_runs(run_id)
        ON DELETE CASCADE,
    vibration NUMERIC(10, 4) NOT NULL CHECK (vibration >= 0),
    temperature NUMERIC(10, 4) NOT NULL,
    operating_hours NUMERIC(12, 2) NOT NULL CHECK (operating_hours >= 0),
    pressure_deviation NUMERIC(10, 4) NOT NULL CHECK (pressure_deviation >= 0),
    actual_label TEXT NOT NULL
        CHECK (actual_label IN ('normal', 'maintenance_required')),
    partition TEXT NOT NULL CHECK (partition IN ('train', 'test')),
    UNIQUE (run_id, observation_id)
);

CREATE INDEX inspection_run_partition_idx
    ON inspection_observations(run_id, partition);

CREATE TABLE forest_trees (
    tree_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES random_forest_runs(run_id)
        ON DELETE CASCADE,
    tree_number INTEGER NOT NULL CHECK (tree_number > 0),
    random_seed BIGINT NOT NULL,
    trained BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (run_id, tree_number),
    UNIQUE (run_id, tree_id)
);

CREATE TABLE bootstrap_membership (
    tree_id BIGINT NOT NULL REFERENCES forest_trees(tree_id)
        ON DELETE CASCADE,
    observation_id BIGINT NOT NULL
        REFERENCES inspection_observations(observation_id)
        ON DELETE CASCADE,
    draw_count INTEGER NOT NULL CHECK (draw_count > 0),
    PRIMARY KEY (tree_id, observation_id)
);

CREATE TABLE tree_feature_importance (
    tree_id BIGINT NOT NULL REFERENCES forest_trees(tree_id)
        ON DELETE CASCADE,
    feature_name TEXT NOT NULL CHECK (
        feature_name IN (
            'vibration',
            'temperature',
            'operating_hours',
            'pressure_deviation'
        )
    ),
    impurity_reduction NUMERIC(16, 8) NOT NULL
        CHECK (impurity_reduction >= 0),
    PRIMARY KEY (tree_id, feature_name)
);

CREATE TABLE oob_predictions (
    tree_id BIGINT NOT NULL REFERENCES forest_trees(tree_id)
        ON DELETE CASCADE,
    observation_id BIGINT NOT NULL
        REFERENCES inspection_observations(observation_id)
        ON DELETE CASCADE,
    predicted_label TEXT NOT NULL
        CHECK (predicted_label IN ('normal', 'maintenance_required')),
    PRIMARY KEY (tree_id, observation_id)
);

CREATE INDEX oob_observation_idx
    ON oob_predictions(observation_id);

-- OOB predictions are valid only for training observations that were
-- excluded from the corresponding tree's bootstrap sample.
CREATE OR REPLACE FUNCTION validate_oob_membership()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    selected_count INTEGER;
    observation_partition TEXT;
    tree_run BIGINT;
    observation_run BIGINT;
BEGIN
    SELECT draw_count
    INTO selected_count
    FROM bootstrap_membership
    WHERE tree_id = NEW.tree_id
      AND observation_id = NEW.observation_id;

    IF FOUND THEN
        RAISE EXCEPTION
            'Observation % was bootstrapped into tree %',
            NEW.observation_id, NEW.tree_id;
    END IF;

    SELECT t.run_id
    INTO tree_run
    FROM forest_trees t
    WHERE t.tree_id = NEW.tree_id;

    SELECT o.run_id, o.partition
    INTO observation_run, observation_partition
    FROM inspection_observations o
    WHERE o.observation_id = NEW.observation_id;

    IF tree_run IS DISTINCT FROM observation_run THEN
        RAISE EXCEPTION 'Tree and observation belong to different runs';
    END IF;

    IF observation_partition <> 'train' THEN
        RAISE EXCEPTION 'OOB evaluation requires a training observation';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER oob_membership_guard
BEFORE INSERT OR UPDATE ON oob_predictions
FOR EACH ROW EXECUTE FUNCTION validate_oob_membership();

-- Create one experimental run.
INSERT INTO random_forest_runs (
    model_name, random_seed, tree_count, max_depth,
    feature_subset_size, training_fraction, status
)
VALUES (
    'equipment-maintenance-rf',
    2026,
    3,
    5,
    2,
    0.8,
    'training'
);

INSERT INTO inspection_observations (
    run_id, vibration, temperature, operating_hours,
    pressure_deviation, actual_label, partition
)
VALUES
    (1, 1.2, 43.0, 1200, 0.8, 'normal', 'train'),
    (1, 2.0, 51.0, 2200, 1.0, 'normal', 'train'),
    (1, 3.1, 59.0, 3100, 1.7, 'normal', 'train'),
    (1, 7.9, 91.0, 8700, 5.7, 'maintenance_required', 'train'),
    (1, 8.6, 102.0, 11000, 7.0, 'maintenance_required', 'train'),
    (1, 6.8, 83.0, 7600, 4.8, 'maintenance_required', 'train'),
    (1, 1.8, 49.0, 1800, 0.9, 'normal', 'test'),
    (1, 8.1, 98.0, 9800, 6.3, 'maintenance_required', 'test');

INSERT INTO forest_trees (run_id, tree_number, random_seed, trained)
VALUES
    (1, 1, 1001, TRUE),
    (1, 2, 1002, TRUE),
    (1, 3, 1003, TRUE);

-- draw_count records multiplicity. An observation drawn three times has
-- greater influence on that tree's training set than a once-drawn row.
INSERT INTO bootstrap_membership (tree_id, observation_id, draw_count)
VALUES
    (1, 1, 2), (1, 2, 1), (1, 4, 2), (1, 5, 1), (1, 6, 1),
    (2, 1, 1), (2, 3, 2), (2, 4, 1), (2, 5, 2), (2, 6, 1),
    (3, 2, 2), (3, 3, 1), (3, 4, 1), (3, 5, 1), (3, 6, 2);

INSERT INTO tree_feature_importance (tree_id, feature_name, impurity_reduction)
VALUES
    (1, 'vibration', 0.42),
    (1, 'temperature', 0.23),
    (1, 'operating_hours', 0.08),
    (1, 'pressure_deviation', 0.31),
    (2, 'vibration', 0.25),
    (2, 'temperature', 0.30),
    (2, 'operating_hours', 0.12),
    (2, 'pressure_deviation', 0.37),
    (3, 'vibration', 0.38),
    (3, 'temperature', 0.24),
    (3, 'operating_hours', 0.11),
    (3, 'pressure_deviation', 0.31);

-- Each inserted prediction references a training row absent from that
-- tree's bootstrap sample.
INSERT INTO oob_predictions (tree_id, observation_id, predicted_label)
VALUES
    (1, 3, 'normal'),
    (2, 2, 'normal'),
    (3, 1, 'normal');

CREATE VIEW random_forest_run_metrics AS
SELECT
    r.run_id,
    r.model_name,
    r.tree_count,
    COUNT(DISTINCT o.observation_id) FILTER (
        WHERE o.partition = 'train'
    ) AS training_rows,
    COUNT(DISTINCT o.observation_id) FILTER (
        WHERE o.partition = 'test'
    ) AS testing_rows,
    COUNT(DISTINCT p.observation_id) AS oob_evaluated_rows,
    AVG(
        CASE WHEN p.predicted_label = o.actual_label
             THEN 1.0 ELSE 0.0 END
    ) AS oob_accuracy
FROM random_forest_runs r
LEFT JOIN inspection_observations o ON o.run_id = r.run_id
LEFT JOIN oob_predictions p
    ON p.observation_id = o.observation_id
   AND EXISTS (
       SELECT 1 FROM forest_trees t
       WHERE t.tree_id = p.tree_id AND t.run_id = r.run_id
   )
GROUP BY r.run_id, r.model_name, r.tree_count;

CREATE VIEW random_forest_feature_ranking AS
WITH raw_importance AS (
    SELECT
        t.run_id,
        f.feature_name,
        SUM(f.impurity_reduction) AS total_reduction
    FROM tree_feature_importance f
    JOIN forest_trees t ON t.tree_id = f.tree_id
    GROUP BY t.run_id, f.feature_name
)
SELECT
    run_id,
    feature_name,
    total_reduction,
    total_reduction / NULLIF(
        SUM(total_reduction) OVER (PARTITION BY run_id),
        0
    ) AS normalized_importance
FROM raw_importance;

-- Review class distribution before interpreting accuracy. A high accuracy
-- can hide poor recall when one class dominates.
SELECT partition, actual_label, COUNT(*) AS observation_count
FROM inspection_observations
WHERE run_id = 1
GROUP BY partition, actual_label
ORDER BY partition, actual_label;

SELECT *
FROM random_forest_run_metrics
WHERE run_id = 1;

SELECT feature_name, normalized_importance
FROM random_forest_feature_ranking
WHERE run_id = 1
ORDER BY normalized_importance DESC;

-- Aggregate OOB votes per observation. The deterministic tie rule prefers
-- maintenance_required to reduce the risk of silently missing a fault.
WITH vote_counts AS (
    SELECT
        p.observation_id,
        COUNT(*) FILTER (
            WHERE p.predicted_label = 'maintenance_required'
        ) AS maintenance_votes,
        COUNT(*) FILTER (
            WHERE p.predicted_label = 'normal'
        ) AS normal_votes
    FROM oob_predictions p
    JOIN forest_trees t ON t.tree_id = p.tree_id
    WHERE t.run_id = 1
    GROUP BY p.observation_id
),
oob_consensus AS (
    SELECT
        observation_id,
        CASE
            WHEN maintenance_votes >= normal_votes
                THEN 'maintenance_required'
            ELSE 'normal'
        END AS predicted_label
    FROM vote_counts
)
SELECT
    o.observation_id,
    o.actual_label,
    c.predicted_label,
    c.predicted_label = o.actual_label AS correct
FROM oob_consensus c
JOIN inspection_observations o
    ON o.observation_id = c.observation_id
ORDER BY o.observation_id;

UPDATE random_forest_runs
SET status = 'completed'
WHERE run_id = 1
  AND EXISTS (
      SELECT 1
      FROM forest_trees t
      WHERE t.run_id = random_forest_runs.run_id
        AND t.trained
  );

COMMIT;
