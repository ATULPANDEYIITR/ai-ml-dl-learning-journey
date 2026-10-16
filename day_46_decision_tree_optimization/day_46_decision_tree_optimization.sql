-- PostgreSQL 15+
-- Decision tree optimization experiment registry.
--
-- This schema records model configurations, training runs, evaluation metrics,
-- feature importance, and selected pruning policies. It separates model
-- complexity settings from evaluation evidence and records the data partition
-- used for each metric to help prevent accidental test-set tuning.

BEGIN;

DROP VIEW IF EXISTS decision_tree_experiment_report;
DROP TABLE IF EXISTS feature_importance CASCADE;
DROP TABLE IF EXISTS evaluation_metrics CASCADE;
DROP TABLE IF EXISTS model_runs CASCADE;
DROP TABLE IF EXISTS dataset_versions CASCADE;
DROP TABLE IF EXISTS decision_tree_policies CASCADE;
DROP TABLE IF EXISTS model_registry CASCADE;

CREATE TABLE model_registry (
    model_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_name TEXT NOT NULL UNIQUE,
    task_name TEXT NOT NULL,
    target_column TEXT NOT NULL,
    criterion TEXT NOT NULL CHECK (criterion IN ('gini', 'entropy')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    active BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE UNIQUE INDEX one_active_model_per_task
    ON model_registry (task_name)
    WHERE active;

CREATE TABLE decision_tree_policies (
    policy_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    policy_name TEXT NOT NULL UNIQUE,
    max_depth INTEGER,
    min_samples_split INTEGER NOT NULL DEFAULT 2,
    min_samples_leaf INTEGER NOT NULL DEFAULT 1,
    min_impurity_decrease NUMERIC(12, 8) NOT NULL DEFAULT 0,
    pruning_method TEXT NOT NULL CHECK (
        pruning_method IN ('none', 'pre_pruning', 'cost_complexity')
    ),
    pruning_alpha NUMERIC(12, 8) NOT NULL DEFAULT 0,
    require_validation BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CHECK (max_depth IS NULL OR max_depth >= 0),
    CHECK (min_samples_split >= 2),
    CHECK (min_samples_leaf >= 1),
    CHECK (min_impurity_decrease >= 0),
    CHECK (pruning_alpha >= 0),
    CHECK (
        pruning_method = 'cost_complexity'
        OR pruning_alpha = 0
    )
);

CREATE TABLE dataset_versions (
    dataset_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_name TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    target_name TEXT NOT NULL,
    row_count BIGINT NOT NULL CHECK (row_count > 0),
    feature_count INTEGER NOT NULL CHECK (feature_count > 0),
    label_distribution JSONB NOT NULL,
    source_fingerprint TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (dataset_name, dataset_version),
    CHECK (jsonb_typeof(label_distribution) = 'object')
);

CREATE TABLE model_runs (
    run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_id BIGINT NOT NULL REFERENCES model_registry(model_id),
    policy_id BIGINT NOT NULL REFERENCES decision_tree_policies(policy_id),
    dataset_id BIGINT NOT NULL REFERENCES dataset_versions(dataset_id),
    run_status TEXT NOT NULL CHECK (
        run_status IN ('queued', 'running', 'completed', 'failed')
    ),
    random_seed INTEGER NOT NULL,
    actual_depth INTEGER CHECK (actual_depth IS NULL OR actual_depth >= 0),
    node_count INTEGER CHECK (node_count IS NULL OR node_count >= 1),
    leaf_count INTEGER CHECK (leaf_count IS NULL OR leaf_count >= 1),
    training_started_at TIMESTAMPTZ,
    training_finished_at TIMESTAMPTZ,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CHECK (
        training_finished_at IS NULL
        OR training_started_at IS NULL
        OR training_finished_at >= training_started_at
    ),
    CHECK (
        leaf_count IS NULL
        OR node_count IS NULL
        OR leaf_count <= node_count
    ),
    CHECK (
        run_status <> 'completed'
        OR (
            actual_depth IS NOT NULL
            AND node_count IS NOT NULL
            AND leaf_count IS NOT NULL
            AND training_finished_at IS NOT NULL
        )
    ),
    CHECK (
        run_status <> 'failed'
        OR error_message IS NOT NULL
    )
);

CREATE INDEX model_runs_by_model_and_time
    ON model_runs (model_id, created_at DESC);

CREATE INDEX model_runs_by_status
    ON model_runs (run_status, created_at DESC);

CREATE TABLE evaluation_metrics (
    metric_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES model_runs(run_id) ON DELETE CASCADE,
    partition_name TEXT NOT NULL CHECK (
        partition_name IN ('training', 'validation', 'test')
    ),
    sample_count INTEGER NOT NULL CHECK (sample_count > 0),
    accuracy NUMERIC(8, 7) NOT NULL CHECK (accuracy BETWEEN 0 AND 1),
    precision_score NUMERIC(8, 7) CHECK (
        precision_score IS NULL OR precision_score BETWEEN 0 AND 1
    ),
    recall_score NUMERIC(8, 7) CHECK (
        recall_score IS NULL OR recall_score BETWEEN 0 AND 1
    ),
    f1_score NUMERIC(8, 7) CHECK (
        f1_score IS NULL OR f1_score BETWEEN 0 AND 1
    ),
    false_negative_count INTEGER NOT NULL DEFAULT 0 CHECK (
        false_negative_count >= 0
    ),
    false_positive_count INTEGER NOT NULL DEFAULT 0 CHECK (
        false_positive_count >= 0
    ),
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (run_id, partition_name)
);

CREATE INDEX evaluation_metrics_partition_accuracy
    ON evaluation_metrics (partition_name, accuracy DESC);

CREATE TABLE feature_importance (
    run_id BIGINT NOT NULL REFERENCES model_runs(run_id) ON DELETE CASCADE,
    feature_name TEXT NOT NULL,
    importance NUMERIC(12, 10) NOT NULL CHECK (
        importance BETWEEN 0 AND 1
    ),
    split_count INTEGER NOT NULL CHECK (split_count >= 0),
    PRIMARY KEY (run_id, feature_name)
);

CREATE OR REPLACE FUNCTION validate_completed_tree_run()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    dataset_features INTEGER;
    configured_depth INTEGER;
    configured_pruning TEXT;
BEGIN
    IF NEW.run_status = 'completed' THEN
        SELECT feature_count
        INTO dataset_features
        FROM dataset_versions
        WHERE dataset_id = NEW.dataset_id;

        SELECT max_depth, pruning_method
        INTO configured_depth, configured_pruning
        FROM decision_tree_policies
        WHERE policy_id = NEW.policy_id;

        IF NEW.training_started_at IS NULL OR NEW.training_finished_at IS NULL THEN
            RAISE EXCEPTION
                'A completed run requires start and finish timestamps';
        END IF;

        IF configured_depth IS NOT NULL AND NEW.actual_depth > configured_depth THEN
            RAISE EXCEPTION
                'Actual tree depth % exceeds configured maximum %',
                NEW.actual_depth, configured_depth;
        END IF;

        IF NEW.leaf_count > NEW.node_count THEN
            RAISE EXCEPTION 'Leaf count cannot exceed node count';
        END IF;

        IF dataset_features IS NULL OR dataset_features <= 0 THEN
            RAISE EXCEPTION 'The dataset has no valid feature definition';
        END IF;

        IF configured_pruning = 'cost_complexity' THEN
            PERFORM 1
            FROM decision_tree_policies
            WHERE policy_id = NEW.policy_id
              AND pruning_alpha >= 0;

            IF NOT FOUND THEN
                RAISE EXCEPTION 'Invalid cost-complexity policy';
            END IF;
        END IF;
    END IF;

    IF NEW.run_status = 'failed' AND NEW.error_message IS NULL THEN
        RAISE EXCEPTION 'Failed runs require an error message';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER model_run_integrity
BEFORE INSERT OR UPDATE ON model_runs
FOR EACH ROW
EXECUTE FUNCTION validate_completed_tree_run();

CREATE OR REPLACE FUNCTION validate_feature_importance_total()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    affected_run BIGINT;
    total_importance NUMERIC;
BEGIN
    affected_run := COALESCE(NEW.run_id, OLD.run_id);

    SELECT COALESCE(SUM(importance), 0)
    INTO total_importance
    FROM feature_importance
    WHERE run_id = affected_run;

    -- Individual inserts may temporarily make the total less than one.
    -- Reject only totals that exceed one beyond numerical tolerance.
    IF total_importance > 1.000001 THEN
        RAISE EXCEPTION
            'Feature importance total for run % exceeds one: %',
            affected_run, total_importance;
    END IF;

    RETURN COALESCE(NEW, OLD);
END;
$$;

CREATE TRIGGER feature_importance_total_guard
AFTER INSERT OR UPDATE OR DELETE ON feature_importance
FOR EACH ROW
EXECUTE FUNCTION validate_feature_importance_total();

INSERT INTO model_registry (
    model_name, task_name, target_column, criterion, active
)
VALUES
    ('maintenance_tree_gini', 'equipment_failure', 'failure_state', 'gini', TRUE),
    ('maintenance_tree_entropy', 'equipment_failure_experiment', 'failure_state', 'entropy', FALSE);

INSERT INTO decision_tree_policies (
    policy_name, max_depth, min_samples_split, min_samples_leaf,
    min_impurity_decrease, pruning_method, pruning_alpha
)
VALUES
    ('unrestricted_baseline', NULL, 2, 1, 0, 'none', 0),
    ('depth_limited', 4, 8, 4, 0.002, 'pre_pruning', 0),
    ('cost_complexity_alpha_0012', NULL, 4, 2, 0, 'cost_complexity', 0.012);

INSERT INTO dataset_versions (
    dataset_name, dataset_version, target_name, row_count,
    feature_count, label_distribution, source_fingerprint
)
VALUES
    (
        'industrial_maintenance',
        '2026.10.synthetic-a',
        'failure_state',
        320,
        4,
        '{"normal": 208, "inspection_required": 112}'::jsonb,
        'sha256:synthetic-maintenance-seed-20261016'
    );

-- Completed baseline run.
INSERT INTO model_runs (
    model_id, policy_id, dataset_id, run_status, random_seed,
    actual_depth, node_count, leaf_count,
    training_started_at, training_finished_at
)
SELECT m.model_id, p.policy_id, d.dataset_id, 'completed', 2026,
       8, 39, 20,
       CURRENT_TIMESTAMP - INTERVAL '5 minutes',
       CURRENT_TIMESTAMP
FROM model_registry m
JOIN decision_tree_policies p
    ON p.policy_name = 'unrestricted_baseline'
JOIN dataset_versions d
    ON d.dataset_name = 'industrial_maintenance'
WHERE m.model_name = 'maintenance_tree_gini';

-- Pre-pruned run with smaller model complexity.
INSERT INTO model_runs (
    model_id, policy_id, dataset_id, run_status, random_seed,
    actual_depth, node_count, leaf_count,
    training_started_at, training_finished_at
)
SELECT m.model_id, p.policy_id, d.dataset_id, 'completed', 2026,
       4, 17, 9,
       CURRENT_TIMESTAMP - INTERVAL '4 minutes',
       CURRENT_TIMESTAMP
FROM model_registry m
JOIN decision_tree_policies p
    ON p.policy_name = 'depth_limited'
JOIN dataset_versions d
    ON d.dataset_name = 'industrial_maintenance'
WHERE m.model_name = 'maintenance_tree_gini';

INSERT INTO evaluation_metrics (
    run_id, partition_name, sample_count, accuracy,
    precision_score, recall_score, f1_score,
    false_negative_count, false_positive_count
)
SELECT run_id, 'training', 256, 0.9921875, 0.99, 0.98, 0.985,
       1, 1
FROM model_runs
WHERE policy_id = (
    SELECT policy_id FROM decision_tree_policies
    WHERE policy_name = 'unrestricted_baseline'
);

INSERT INTO evaluation_metrics (
    run_id, partition_name, sample_count, accuracy,
    precision_score, recall_score, f1_score,
    false_negative_count, false_positive_count
)
SELECT run_id, 'validation', 64, 0.828125, 0.80, 0.75, 0.7741935,
       4, 7
FROM model_runs
WHERE policy_id = (
    SELECT policy_id FROM decision_tree_policies
    WHERE policy_name = 'unrestricted_baseline'
);

INSERT INTO evaluation_metrics (
    run_id, partition_name, sample_count, accuracy,
    precision_score, recall_score, f1_score,
    false_negative_count, false_positive_count
)
SELECT run_id, 'training', 256, 0.9140625, 0.88, 0.86, 0.8695652,
       9, 13
FROM model_runs
WHERE policy_id = (
    SELECT policy_id FROM decision_tree_policies
    WHERE policy_name = 'depth_limited'
);

INSERT INTO evaluation_metrics (
    run_id, partition_name, sample_count, accuracy,
    precision_score, recall_score, f1_score,
    false_negative_count, false_positive_count
)
SELECT run_id, 'validation', 64, 0.890625, 0.87, 0.875, 0.872,
       2, 4
FROM model_runs
WHERE policy_id = (
    SELECT policy_id FROM decision_tree_policies
    WHERE policy_name = 'depth_limited'
);

INSERT INTO feature_importance (run_id, feature_name, importance, split_count)
SELECT run_id, 'vibration_mm_s', 0.48, 8
FROM model_runs
WHERE policy_id = (
    SELECT policy_id FROM decision_tree_policies
    WHERE policy_name = 'depth_limited'
);

INSERT INTO feature_importance (run_id, feature_name, importance, split_count)
SELECT run_id, 'temperature_c', 0.27, 5
FROM model_runs
WHERE policy_id = (
    SELECT policy_id FROM decision_tree_policies
    WHERE policy_name = 'depth_limited'
);

INSERT INTO feature_importance (run_id, feature_name, importance, split_count)
SELECT run_id, 'operating_hours', 0.16, 3
FROM model_runs
WHERE policy_id = (
    SELECT policy_id FROM decision_tree_policies
    WHERE policy_name = 'depth_limited'
);

INSERT INTO feature_importance (run_id, feature_name, importance, split_count)
SELECT run_id, 'previous_fault_count', 0.09, 1
FROM model_runs
WHERE policy_id = (
    SELECT policy_id FROM decision_tree_policies
    WHERE policy_name = 'depth_limited'
);

CREATE VIEW decision_tree_experiment_report AS
SELECT
    r.run_id,
    m.model_name,
    p.policy_name,
    p.max_depth AS configured_max_depth,
    p.pruning_method,
    p.pruning_alpha,
    d.dataset_name,
    d.dataset_version,
    r.run_status,
    r.actual_depth,
    r.node_count,
    r.leaf_count,
    e.partition_name,
    e.sample_count,
    e.accuracy,
    e.precision_score,
    e.recall_score,
    e.f1_score,
    e.false_negative_count,
    e.false_positive_count,
    r.created_at
FROM model_runs r
JOIN model_registry m ON m.model_id = r.model_id
JOIN decision_tree_policies p ON p.policy_id = r.policy_id
JOIN dataset_versions d ON d.dataset_id = r.dataset_id
LEFT JOIN evaluation_metrics e ON e.run_id = r.run_id;

-- Compare training and validation performance without mixing partitions.
SELECT
    model_name,
    policy_name,
    MAX(accuracy) FILTER (WHERE partition_name = 'training') AS training_accuracy,
    MAX(accuracy) FILTER (WHERE partition_name = 'validation') AS validation_accuracy,
    MAX(false_negative_count)
        FILTER (WHERE partition_name = 'validation') AS validation_false_negatives,
    MAX(actual_depth) AS actual_depth,
    MAX(leaf_count) AS leaf_count
FROM decision_tree_experiment_report
WHERE run_status = 'completed'
GROUP BY model_name, policy_name
ORDER BY validation_accuracy DESC NULLS LAST, leaf_count ASC NULLS LAST;

-- Rank features within a model run. These values describe split-based
-- predictive importance and must not be interpreted as causal effects.
SELECT
    r.run_id,
    p.policy_name,
    fi.feature_name,
    fi.importance,
    fi.split_count,
    RANK() OVER (
        PARTITION BY r.run_id
        ORDER BY fi.importance DESC
    ) AS importance_rank
FROM feature_importance fi
JOIN model_runs r ON r.run_id = fi.run_id
JOIN decision_tree_policies p ON p.policy_id = r.policy_id
ORDER BY r.run_id, importance_rank;

-- Select the best validation run, preferring fewer leaves when accuracy ties.
WITH validation_results AS (
    SELECT
        r.run_id,
        p.policy_name,
        r.leaf_count,
        e.accuracy,
        e.recall_score,
        e.false_negative_count
    FROM model_runs r
    JOIN decision_tree_policies p ON p.policy_id = r.policy_id
    JOIN evaluation_metrics e ON e.run_id = r.run_id
    WHERE r.run_status = 'completed'
      AND e.partition_name = 'validation'
)
SELECT *
FROM validation_results
ORDER BY accuracy DESC, leaf_count ASC, false_negative_count ASC
LIMIT 1;

-- Database integrity examples. The following statements are intentionally
-- disabled because they must raise constraint errors if executed.
-- INSERT INTO decision_tree_policies
--     (policy_name, min_samples_split, min_samples_leaf,
--      min_impurity_decrease, pruning_method, pruning_alpha)
-- VALUES ('invalid_leaf_size', 2, 0, 0, 'none', 0);
--
-- INSERT INTO feature_importance (run_id, feature_name, importance, split_count)
-- VALUES (2, 'invalid_feature', 1.5, 3);

COMMIT;
