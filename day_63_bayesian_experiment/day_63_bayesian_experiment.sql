DROP SCHEMA IF EXISTS bayesian_optimization CASCADE;
CREATE SCHEMA bayesian_optimization;

SET search_path TO bayesian_optimization, public;

CREATE TYPE optimization_direction AS ENUM (
    'maximize',
    'minimize'
);

CREATE TYPE experiment_status AS ENUM (
    'draft',
    'running',
    'paused',
    'completed',
    'failed'
);

CREATE TYPE observation_status AS ENUM (
    'accepted',
    'rejected',
    'failed'
);

CREATE TABLE experiments (
    experiment_id BIGSERIAL PRIMARY KEY,
    experiment_name TEXT NOT NULL UNIQUE,
    objective_name TEXT NOT NULL,
    direction optimization_direction NOT NULL,
    status experiment_status NOT NULL DEFAULT 'draft',
    exploration_weight DOUBLE PRECISION NOT NULL DEFAULT 0.01,
    noise_variance DOUBLE PRECISION NOT NULL DEFAULT 0.00001,
    length_scale DOUBLE PRECISION NOT NULL DEFAULT 0.8,
    signal_variance DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    max_evaluations INTEGER NOT NULL DEFAULT 50,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    CHECK (length_scale > 0),
    CHECK (signal_variance > 0),
    CHECK (noise_variance >= 0),
    CHECK (exploration_weight >= 0),
    CHECK (max_evaluations > 0),
    CHECK (
        (status = 'completed' AND completed_at IS NOT NULL)
        OR status <> 'completed'
    )
);

CREATE TABLE dimensions (
    dimension_id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,
    dimension_name TEXT NOT NULL,
    lower_bound DOUBLE PRECISION NOT NULL,
    upper_bound DOUBLE PRECISION NOT NULL,
    CHECK (lower_bound < upper_bound),
    UNIQUE (experiment_id, dimension_name)
);

CREATE TABLE acquisition_strategies (
    strategy_id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,
    strategy_name TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    parameter_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (
        experiment_id,
        strategy_name,
        strategy_version
    )
);

CREATE TABLE observations (
    observation_id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,
    evaluation_number INTEGER NOT NULL,
    x_values JSONB NOT NULL,
    objective_value DOUBLE PRECISION,
    status observation_status NOT NULL DEFAULT 'accepted',
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (experiment_id, evaluation_number),
    CHECK (jsonb_typeof(x_values) = 'array'),
    CHECK (
        status <> 'accepted'
        OR objective_value IS NOT NULL
    )
);

CREATE TABLE surrogate_predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,
    observation_id BIGINT
        REFERENCES observations(observation_id)
        ON DELETE SET NULL,
    x_values JSONB NOT NULL,
    predicted_mean DOUBLE PRECISION NOT NULL,
    predicted_stddev DOUBLE PRECISION NOT NULL,
    expected_improvement DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (predicted_stddev >= 0),
    CHECK (expected_improvement >= 0)
);

CREATE TABLE optimization_decisions (
    decision_id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,
    evaluation_number INTEGER NOT NULL,
    selected_x JSONB NOT NULL,
    acquisition_value DOUBLE PRECISION NOT NULL,
    decision_reason TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (acquisition_value >= 0)
);

CREATE INDEX idx_observations_experiment_status
    ON observations (experiment_id, status);

CREATE INDEX idx_observations_experiment_value
    ON observations (experiment_id, objective_value DESC)
    WHERE status = 'accepted';

CREATE INDEX idx_predictions_experiment_created
    ON surrogate_predictions (experiment_id, created_at DESC);

CREATE INDEX idx_decisions_experiment_evaluation
    ON optimization_decisions (experiment_id, evaluation_number);

CREATE OR REPLACE FUNCTION validate_observation_dimensions()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    expected_count INTEGER;
    actual_count INTEGER;
BEGIN
    SELECT COUNT(*)
    INTO expected_count
    FROM dimensions
    WHERE experiment_id = NEW.experiment_id;

    SELECT jsonb_array_length(NEW.x_values)
    INTO actual_count;

    IF expected_count <> actual_count THEN
        RAISE EXCEPTION
            'Observation has % dimensions, expected %',
            actual_count,
            expected_count;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_observation_dimensions
BEFORE INSERT OR UPDATE ON observations
FOR EACH ROW
EXECUTE FUNCTION validate_observation_dimensions();

CREATE OR REPLACE FUNCTION validate_completed_experiment()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    accepted_count INTEGER;
BEGIN
    IF NEW.status = 'completed' THEN
        SELECT COUNT(*)
        INTO accepted_count
        FROM observations
        WHERE experiment_id = NEW.experiment_id
          AND status = 'accepted';

        IF accepted_count = 0 THEN
            RAISE EXCEPTION
                'Experiment cannot be completed without accepted observations';
        END IF;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_completed_experiment
BEFORE UPDATE ON experiments
FOR EACH ROW
EXECUTE FUNCTION validate_completed_experiment();

CREATE OR REPLACE FUNCTION best_observation(
    p_experiment_id BIGINT
)
RETURNS TABLE (
    observation_id BIGINT,
    evaluation_number INTEGER,
    objective_value DOUBLE PRECISION,
    x_values JSONB
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        o.observation_id,
        o.evaluation_number,
        o.objective_value,
        o.x_values
    FROM observations AS o
    JOIN experiments AS e
      ON e.experiment_id = o.experiment_id
    WHERE o.experiment_id = p_experiment_id
      AND o.status = 'accepted'
    ORDER BY
        CASE
            WHEN e.direction = 'maximize'
                THEN o.objective_value
        END DESC,
        CASE
            WHEN e.direction = 'minimize'
                THEN o.objective_value
        END ASC
    LIMIT 1;
$$;

CREATE OR REPLACE FUNCTION experiment_evaluation_count(
    p_experiment_id BIGINT
)
RETURNS INTEGER
LANGUAGE sql
STABLE
AS $$
    SELECT COUNT(*)::INTEGER
    FROM observations
    WHERE experiment_id = p_experiment_id
      AND status = 'accepted';
$$;

CREATE OR REPLACE FUNCTION evaluate_merge_candidate(
    p_experiment_id BIGINT,
    p_candidate JSONB
)
RETURNS TABLE (
    candidate JSONB,
    within_domain BOOLEAN,
    dimensions_valid BOOLEAN
)
LANGUAGE plpgsql
AS $$
DECLARE
    dimension_row RECORD;
    index_position INTEGER := 0;
    candidate_value DOUBLE PRECISION;
BEGIN
    IF jsonb_typeof(p_candidate) <> 'array' THEN
        RAISE EXCEPTION 'Candidate must be a JSON array';
    END IF;

    IF jsonb_array_length(p_candidate) <>
       (SELECT COUNT(*)
        FROM dimensions
        WHERE experiment_id = p_experiment_id) THEN
        RETURN QUERY
        SELECT p_candidate, FALSE, FALSE;
        RETURN;
    END IF;

    FOR dimension_row IN
        SELECT *
        FROM dimensions
        WHERE experiment_id = p_experiment_id
        ORDER BY dimension_id
    LOOP
        candidate_value :=
            (p_candidate -> index_position)::text::double precision;

        IF candidate_value < dimension_row.lower_bound
           OR candidate_value > dimension_row.upper_bound THEN
            RETURN QUERY
            SELECT p_candidate, FALSE, TRUE;
            RETURN;
        END IF;

        index_position := index_position + 1;
    END LOOP;

    RETURN QUERY
    SELECT p_candidate, TRUE, TRUE;
END;
$$;

INSERT INTO experiments (
    experiment_name,
    objective_name,
    direction,
    status,
    exploration_weight,
    noise_variance,
    length_scale,
    signal_variance,
    max_evaluations
)
VALUES (
    'pricing_policy_surface',
    'risk_adjusted_conversion',
    'maximize',
    'running',
    0.01,
    0.00001,
    0.8,
    1.0,
    40
);

INSERT INTO dimensions (
    experiment_id,
    dimension_name,
    lower_bound,
    upper_bound
)
SELECT
    experiment_id,
    'price_adjustment',
    -4.0,
    4.0
FROM experiments
WHERE experiment_name = 'pricing_policy_surface';

INSERT INTO dimensions (
    experiment_id,
    dimension_name,
    lower_bound,
    upper_bound
)
SELECT
    experiment_id,
    'inventory_pressure',
    -4.0,
    4.0
FROM experiments
WHERE experiment_name = 'pricing_policy_surface';

INSERT INTO acquisition_strategies (
    experiment_id,
    strategy_name,
    strategy_version,
    parameter_json
)
SELECT
    experiment_id,
    'expected_improvement',
    '1.0',
    jsonb_build_object(
        'exploration_weight', 0.01,
        'surrogate', 'gaussian_process',
        'kernel', 'rbf'
    )
FROM experiments
WHERE experiment_name = 'pricing_policy_surface';

INSERT INTO observations (
    experiment_id,
    evaluation_number,
    x_values,
    objective_value,
    status
)
SELECT
    experiment_id,
    1,
    '[-3.0, -2.5]'::jsonb,
    0.115,
    'accepted'
FROM experiments
WHERE experiment_name = 'pricing_policy_surface';

INSERT INTO observations (
    experiment_id,
    evaluation_number,
    x_values,
    objective_value,
    status
)
SELECT
    experiment_id,
    2,
    '[-1.0, 1.0]'::jsonb,
    0.420,
    'accepted'
FROM experiments
WHERE experiment_name = 'pricing_policy_surface';

INSERT INTO observations (
    experiment_id,
    evaluation_number,
    x_values,
    objective_value,
    status
)
SELECT
    experiment_id,
    3,
    '[1.0, 2.0]'::jsonb,
    1.610,
    'accepted'
FROM experiments
WHERE experiment_name = 'pricing_policy_surface';

INSERT INTO observations (
    experiment_id,
    evaluation_number,
    x_values,
    objective_value,
    status
)
SELECT
    experiment_id,
    4,
    '[2.0, 2.0]'::jsonb,
    1.720,
    'accepted'
FROM experiments
WHERE experiment_name = 'pricing_policy_surface';

INSERT INTO observations (
    experiment_id,
    evaluation_number,
    x_values,
    objective_value,
    status
)
SELECT
    experiment_id,
    5,
    '[3.5, 3.5]'::jsonb,
    0.250,
    'accepted'
FROM experiments
WHERE experiment_name = 'pricing_policy_surface';

INSERT INTO observations (
    experiment_id,
    evaluation_number,
    x_values,
    objective_value,
    status
)
SELECT
    experiment_id,
    6,
    '[0.0, 0.0]'::jsonb,
    NULL,
    'failed'
FROM experiments
WHERE experiment_name = 'pricing_policy_surface';

INSERT INTO optimization_decisions (
    experiment_id,
    evaluation_number,
    selected_x,
    acquisition_value,
    decision_reason
)
SELECT
    experiment_id,
    7,
    '[1.5, 2.0]'::jsonb,
    0.312,
    'High expected improvement from surrogate mean and predictive uncertainty'
FROM experiments
WHERE experiment_name = 'pricing_policy_surface';

INSERT INTO surrogate_predictions (
    experiment_id,
    observation_id,
    x_values,
    predicted_mean,
    predicted_stddev,
    expected_improvement
)
SELECT
    e.experiment_id,
    NULL,
    '[1.5, 2.0]'::jsonb,
    1.680,
    0.120,
    0.312
FROM experiments AS e
WHERE e.experiment_name = 'pricing_policy_surface';

BEGIN;

WITH next_evaluation AS (
    SELECT
        experiment_id,
        COALESCE(
            MAX(evaluation_number),
            0
        ) + 1 AS evaluation_number
    FROM observations
    WHERE experiment_id = (
        SELECT experiment_id
        FROM experiments
        WHERE experiment_name = 'pricing_policy_surface'
    )
    GROUP BY experiment_id
)
INSERT INTO observations (
    experiment_id,
    evaluation_number,
    x_values,
    objective_value,
    status
)
SELECT
    experiment_id,
    evaluation_number,
    '[1.5, 2.0]'::jsonb,
    1.755,
    'accepted'
FROM next_evaluation;

UPDATE experiments
SET status = 'completed',
    completed_at = CURRENT_TIMESTAMP
WHERE experiment_name = 'pricing_policy_surface';

COMMIT;

SELECT
    e.experiment_name,
    e.status,
    e.direction,
    COUNT(o.observation_id) FILTER (
        WHERE o.status = 'accepted'
    ) AS accepted_evaluations,
    COUNT(o.observation_id) FILTER (
        WHERE o.status = 'failed'
    ) AS failed_evaluations
FROM experiments AS e
LEFT JOIN observations AS o
    ON o.experiment_id = e.experiment_id
GROUP BY
    e.experiment_id,
    e.experiment_name,
    e.status,
    e.direction;

SELECT *
FROM best_observation(
    (
        SELECT experiment_id
        FROM experiments
        WHERE experiment_name = 'pricing_policy_surface'
    )
);

WITH observation_statistics AS (
    SELECT
        experiment_id,
        COUNT(*) AS accepted_count,
        AVG(objective_value) AS mean_objective,
        MIN(objective_value) AS minimum_objective,
        MAX(objective_value) AS maximum_objective,
        STDDEV_POP(objective_value) AS objective_stddev
    FROM observations
    WHERE status = 'accepted'
    GROUP BY experiment_id
)
SELECT
    e.experiment_name,
    s.accepted_count,
    ROUND(s.mean_objective::numeric, 4)
        AS mean_objective,
    ROUND(s.minimum_objective::numeric, 4)
        AS minimum_objective,
    ROUND(s.maximum_objective::numeric, 4)
        AS maximum_objective,
    ROUND(s.objective_stddev::numeric, 4)
        AS objective_stddev
FROM observation_statistics AS s
JOIN experiments AS e
  ON e.experiment_id = s.experiment_id;

SELECT *
FROM evaluate_merge_candidate(
    (
        SELECT experiment_id
        FROM experiments
        WHERE experiment_name = 'pricing_policy_surface'
    ),
    '[1.5, 2.0]'::jsonb
);

SELECT *
FROM evaluate_merge_candidate(
    (
        SELECT experiment_id
        FROM experiments
        WHERE experiment_name = 'pricing_policy_surface'
    ),
    '[9.0, 2.0]'::jsonb
);

SELECT
    d.evaluation_number,
    d.selected_x,
    d.acquisition_value,
    d.decision_reason,
    d.created_at
FROM optimization_decisions AS d
JOIN experiments AS e
  ON e.experiment_id = d.experiment_id
WHERE e.experiment_name = 'pricing_policy_surface'
ORDER BY d.evaluation_number;

SELECT
    e.experiment_name,
    d.dimension_name,
    d.lower_bound,
    d.upper_bound
FROM experiments AS e
JOIN dimensions AS d
  ON d.experiment_id = e.experiment_id
WHERE e.experiment_name = 'pricing_policy_surface'
ORDER BY d.dimension_id;
