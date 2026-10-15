-- PostgreSQL 14+
-- Decision-tree training data, candidate splits, impurity measurements,
-- and split selection for software release-risk classification.
--
-- The schema stores the observations and split statistics needed to inspect
-- entropy, Gini impurity, information gain, and the recursive partitioning
-- of a training dataset. SQL computes candidate thresholds and ranks splits.
-- Tree recursion is represented by parent-child node relationships.

BEGIN;

DROP SCHEMA IF EXISTS decision_tree_lab CASCADE;
CREATE SCHEMA decision_tree_lab;
SET search_path TO decision_tree_lab, public;

CREATE TYPE impurity_criterion AS ENUM ('entropy', 'gini');
CREATE TYPE release_decision AS ENUM ('eligible', 'manual_review');
CREATE TYPE feature_kind AS ENUM ('numeric');

CREATE TABLE training_run (
    run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_name TEXT NOT NULL UNIQUE,
    criterion impurity_criterion NOT NULL,
    max_depth INTEGER NOT NULL CHECK (max_depth >= 0),
    min_samples_leaf INTEGER NOT NULL CHECK (min_samples_leaf >= 1),
    min_gain NUMERIC NOT NULL DEFAULT 0
        CHECK (min_gain >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE training_observation (
    observation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES training_run(run_id)
        ON DELETE CASCADE,
    coverage NUMERIC NOT NULL CHECK (coverage BETWEEN 0 AND 100),
    failed_tests INTEGER NOT NULL CHECK (failed_tests >= 0),
    security_findings INTEGER NOT NULL CHECK (security_findings >= 0),
    decision release_decision NOT NULL,
    CONSTRAINT unique_observation_per_run UNIQUE (run_id, observation_id)
);

CREATE INDEX training_observation_run_idx
    ON training_observation(run_id);

CREATE INDEX training_observation_class_idx
    ON training_observation(run_id, decision);

CREATE TABLE feature_definition (
    feature_name TEXT PRIMARY KEY,
    kind feature_kind NOT NULL,
    description TEXT NOT NULL
);

INSERT INTO feature_definition (feature_name, kind, description) VALUES
    ('coverage', 'numeric',
     'Percentage of required automated tests covered'),
    ('failed_tests', 'numeric',
     'Number of failed automated tests'),
    ('security_findings', 'numeric',
     'Number of unresolved security findings');

CREATE TABLE tree_node (
    node_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES training_run(run_id)
        ON DELETE CASCADE,
    parent_node_id BIGINT,
    depth INTEGER NOT NULL CHECK (depth >= 0),
    sample_count INTEGER NOT NULL CHECK (sample_count > 0),
    eligible_count INTEGER NOT NULL CHECK (eligible_count >= 0),
    manual_review_count INTEGER NOT NULL CHECK (manual_review_count >= 0),
    impurity NUMERIC NOT NULL CHECK (impurity >= 0),
    split_feature TEXT REFERENCES feature_definition(feature_name),
    split_threshold NUMERIC,
    information_gain NUMERIC NOT NULL DEFAULT 0,
    predicted_class release_decision NOT NULL,
    CONSTRAINT node_class_counts_match
        CHECK (sample_count = eligible_count + manual_review_count),
    CONSTRAINT leaf_or_split
        CHECK (
            (split_feature IS NULL AND split_threshold IS NULL)
            OR
            (split_feature IS NOT NULL AND split_threshold IS NOT NULL)
        ),
    CONSTRAINT node_parent_fk
        FOREIGN KEY (parent_node_id)
        REFERENCES tree_node(node_id)
        DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX tree_node_run_depth_idx
    ON tree_node(run_id, depth);

CREATE TABLE node_membership (
    node_id BIGINT NOT NULL REFERENCES tree_node(node_id)
        ON DELETE CASCADE,
    observation_id BIGINT NOT NULL
        REFERENCES training_observation(observation_id)
        ON DELETE CASCADE,
    PRIMARY KEY (node_id, observation_id)
);

CREATE TABLE candidate_split (
    candidate_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES training_run(run_id)
        ON DELETE CASCADE,
    feature_name TEXT NOT NULL
        REFERENCES feature_definition(feature_name),
    threshold NUMERIC NOT NULL,
    parent_sample_count INTEGER NOT NULL CHECK (parent_sample_count > 0),
    left_sample_count INTEGER NOT NULL CHECK (left_sample_count >= 0),
    right_sample_count INTEGER NOT NULL CHECK (right_sample_count >= 0),
    parent_impurity NUMERIC NOT NULL CHECK (parent_impurity >= 0),
    weighted_child_impurity NUMERIC NOT NULL
        CHECK (weighted_child_impurity >= 0),
    gain NUMERIC NOT NULL,
    eligible_left INTEGER NOT NULL CHECK (eligible_left >= 0),
    review_left INTEGER NOT NULL CHECK (review_left >= 0),
    eligible_right INTEGER NOT NULL CHECK (eligible_right >= 0),
    review_right INTEGER NOT NULL CHECK (review_right >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT candidate_partition_counts
        CHECK (
            left_sample_count + right_sample_count = parent_sample_count
        ),
    CONSTRAINT candidate_left_class_counts
        CHECK (
            eligible_left + review_left = left_sample_count
        ),
    CONSTRAINT candidate_right_class_counts
        CHECK (
            eligible_right + review_right = right_sample_count
        ),
    CONSTRAINT candidate_gain_matches
        CHECK (
            abs(gain - (parent_impurity - weighted_child_impurity)) < 0.000001
        ),
    CONSTRAINT candidate_threshold_unique
        UNIQUE (run_id, feature_name, threshold)
);

CREATE INDEX candidate_split_rank_idx
    ON candidate_split(run_id, gain DESC);

-- PostgreSQL's numeric calculations avoid unnecessary floating-point
-- rounding in reported impurity and gain values.
CREATE FUNCTION class_impurity(
    eligible_count INTEGER,
    review_count INTEGER,
    criterion impurity_criterion
)
RETURNS NUMERIC
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    total_count NUMERIC;
    p_eligible NUMERIC;
    p_review NUMERIC;
BEGIN
    IF eligible_count < 0 OR review_count < 0 THEN
        RAISE EXCEPTION 'Class counts cannot be negative';
    END IF;

    total_count := eligible_count + review_count;

    IF total_count = 0 THEN
        RETURN 0;
    END IF;

    p_eligible := eligible_count / total_count;
    p_review := review_count / total_count;

    IF criterion = 'gini' THEN
        RETURN 1 - p_eligible * p_eligible - p_review * p_review;
    END IF;

    RETURN
        CASE WHEN p_eligible = 0 THEN 0
             ELSE -p_eligible * (ln(p_eligible) / ln(2)) END
        +
        CASE WHEN p_review = 0 THEN 0
             ELSE -p_review * (ln(p_review) / ln(2)) END;
END;
$$;

INSERT INTO training_run (
    run_name, criterion, max_depth, min_samples_leaf, min_gain
) VALUES (
    'release-risk-entropy', 'entropy', 4, 2, 0.000001
);

INSERT INTO training_observation (
    run_id, coverage, failed_tests, security_findings, decision
)
SELECT
    run.run_id,
    sample.coverage,
    sample.failed_tests,
    sample.security_findings,
    CASE
        WHEN sample.coverage < 75
          OR sample.failed_tests > 0
          OR sample.security_findings > 0
        THEN 'manual_review'::release_decision
        ELSE 'eligible'::release_decision
    END
FROM training_run AS run
CROSS JOIN (
    VALUES
        (45::NUMERIC, 0, 0),
        (55::NUMERIC, 0, 0),
        (68::NUMERIC, 0, 0),
        (76::NUMERIC, 0, 0),
        (84::NUMERIC, 0, 0),
        (92::NUMERIC, 0, 0),
        (98::NUMERIC, 0, 0),
        (45::NUMERIC, 1, 0),
        (68::NUMERIC, 1, 0),
        (76::NUMERIC, 1, 0),
        (84::NUMERIC, 1, 0),
        (92::NUMERIC, 1, 0),
        (98::NUMERIC, 1, 0),
        (45::NUMERIC, 0, 1),
        (68::NUMERIC, 0, 1),
        (76::NUMERIC, 0, 1),
        (84::NUMERIC, 0, 1),
        (92::NUMERIC, 0, 1),
        (98::NUMERIC, 0, 1),
        (55::NUMERIC, 3, 1),
        (76::NUMERIC, 3, 0),
        (84::NUMERIC, 3, 0),
        (92::NUMERIC, 3, 0),
        (98::NUMERIC, 3, 0)
) AS sample(coverage, failed_tests, security_findings)
WHERE run.run_name = 'release-risk-entropy';

-- The parent impurity is calculated from the complete training set.
CREATE VIEW run_class_distribution AS
SELECT
    run_id,
    COUNT(*) AS sample_count,
    COUNT(*) FILTER (WHERE decision = 'eligible') AS eligible_count,
    COUNT(*) FILTER (WHERE decision = 'manual_review') AS review_count
FROM training_observation
GROUP BY run_id;

CREATE VIEW run_impurity AS
SELECT
    run.run_id,
    run.run_name,
    run.criterion,
    distribution.sample_count,
    distribution.eligible_count,
    distribution.review_count,
    class_impurity(
        distribution.eligible_count,
        distribution.review_count,
        run.criterion
    ) AS parent_impurity
FROM training_run AS run
JOIN run_class_distribution AS distribution USING (run_id);

-- Thresholds are midpoints between consecutive distinct observed values.
-- Each midpoint creates the same partition as any threshold strictly between
-- the adjacent values, while keeping the rule interpretable.
CREATE VIEW candidate_thresholds AS
WITH feature_values AS (
    SELECT run_id, 'coverage'::TEXT AS feature_name, coverage AS feature_value
    FROM training_observation
    UNION ALL
    SELECT run_id, 'failed_tests', failed_tests::NUMERIC
    FROM training_observation
    UNION ALL
    SELECT run_id, 'security_findings', security_findings::NUMERIC
    FROM training_observation
),
ordered_values AS (
    SELECT
        run_id,
        feature_name,
        feature_value,
        LEAD(feature_value) OVER (
            PARTITION BY run_id, feature_name
            ORDER BY feature_value
        ) AS next_value
    FROM (
        SELECT DISTINCT run_id, feature_name, feature_value
        FROM feature_values
    ) AS distinct_values
)
SELECT
    run_id,
    feature_name,
    feature_value + (next_value - feature_value) / 2 AS threshold
FROM ordered_values
WHERE next_value IS NOT NULL;

-- This view evaluates each candidate against the full training set.
-- FILTER aggregates expose class counts on each side of a split.
CREATE VIEW evaluated_candidate_splits AS
WITH expanded AS (
    SELECT
        threshold.run_id,
        threshold.feature_name,
        threshold.threshold,
        observation.observation_id,
        observation.decision,
        CASE threshold.feature_name
            WHEN 'coverage' THEN observation.coverage
            WHEN 'failed_tests' THEN observation.failed_tests::NUMERIC
            WHEN 'security_findings' THEN observation.security_findings::NUMERIC
        END AS feature_value
    FROM candidate_thresholds AS threshold
    JOIN training_observation AS observation USING (run_id)
),
counts AS (
    SELECT
        run_id,
        feature_name,
        threshold,
        COUNT(*) AS parent_count,
        COUNT(*) FILTER (WHERE feature_value <= threshold) AS left_count,
        COUNT(*) FILTER (WHERE feature_value > threshold) AS right_count,
        COUNT(*) FILTER (
            WHERE feature_value <= threshold AND decision = 'eligible'
        ) AS eligible_left,
        COUNT(*) FILTER (
            WHERE feature_value <= threshold AND decision = 'manual_review'
        ) AS review_left,
        COUNT(*) FILTER (
            WHERE feature_value > threshold AND decision = 'eligible'
        ) AS eligible_right,
        COUNT(*) FILTER (
            WHERE feature_value > threshold AND decision = 'manual_review'
        ) AS review_right
    FROM expanded
    GROUP BY run_id, feature_name, threshold
),
impurities AS (
    SELECT
        counts.*,
        run.criterion,
        class_impurity(
            counts.eligible_left,
            counts.review_left,
            run.criterion
        ) AS left_impurity,
        class_impurity(
            counts.eligible_right,
            counts.review_right,
            run.criterion
        ) AS right_impurity,
        parent.parent_impurity
    FROM counts
    JOIN training_run AS run USING (run_id)
    JOIN run_impurity AS parent USING (run_id)
)
SELECT
    run_id,
    feature_name,
    threshold,
    parent_count,
    left_count,
    right_count,
    eligible_left,
    review_left,
    eligible_right,
    review_right,
    parent_impurity,
    (
        left_count::NUMERIC / parent_count * left_impurity
        +
        right_count::NUMERIC / parent_count * right_impurity
    ) AS weighted_child_impurity,
    parent_impurity - (
        left_count::NUMERIC / parent_count * left_impurity
        +
        right_count::NUMERIC / parent_count * right_impurity
    ) AS gain
FROM impurities;

-- Rank candidate splits by gain while enforcing the configured minimum
-- child size. A deterministic tie-breaker produces stable reports.
CREATE VIEW ranked_candidate_splits AS
SELECT
    candidate.*,
    ROW_NUMBER() OVER (
        PARTITION BY candidate.run_id
        ORDER BY
            candidate.gain DESC,
            candidate.feature_name,
            candidate.threshold
    ) AS gain_rank
FROM evaluated_candidate_splits AS candidate
JOIN training_run AS run USING (run_id)
WHERE candidate.left_count >= run.min_samples_leaf
  AND candidate.right_count >= run.min_samples_leaf
  AND candidate.gain > run.min_gain;

-- Inspect root impurity and the strongest candidate splits.
SELECT
    run_name,
    criterion,
    sample_count,
    eligible_count,
    review_count,
    parent_impurity
FROM run_impurity;

SELECT
    feature_name,
    threshold,
    left_count,
    right_count,
    weighted_child_impurity,
    gain,
    gain_rank
FROM ranked_candidate_splits
WHERE run_id = (
    SELECT run_id
    FROM training_run
    WHERE run_name = 'release-risk-entropy'
)
ORDER BY gain_rank
LIMIT 10;

-- Compare the criteria without changing the source observations.
INSERT INTO training_run (
    run_name, criterion, max_depth, min_samples_leaf, min_gain
)
VALUES (
    'release-risk-gini', 'gini', 4, 2, 0.000001
);

INSERT INTO training_observation (
    run_id, coverage, failed_tests, security_findings, decision
)
SELECT
    gini_run.run_id,
    source.coverage,
    source.failed_tests,
    source.security_findings,
    source.decision
FROM training_observation AS source
JOIN training_run AS entropy_run
    ON entropy_run.run_id = source.run_id
   AND entropy_run.run_name = 'release-risk-entropy'
CROSS JOIN training_run AS gini_run
WHERE gini_run.run_name = 'release-risk-gini';

SELECT
    run_name,
    criterion,
    sample_count,
    parent_impurity
FROM run_impurity
ORDER BY run_name;

-- Database-level validation: an invalid observation must violate a CHECK.
-- This block catches the expected failure so the rest of the script remains
-- executable in a single transaction.
DO $$
DECLARE
    target_run BIGINT;
BEGIN
    SELECT run_id INTO target_run
    FROM training_run
    WHERE run_name = 'release-risk-entropy';

    BEGIN
        INSERT INTO training_observation (
            run_id, coverage, failed_tests, security_findings, decision
        ) VALUES (
            target_run, 150, 0, 0, 'eligible'
        );
    EXCEPTION
        WHEN check_violation THEN
            RAISE NOTICE 'Invalid coverage correctly rejected by constraint';
    END;
END;
$$;

COMMIT;
