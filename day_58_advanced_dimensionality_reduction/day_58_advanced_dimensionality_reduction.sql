DROP SCHEMA IF EXISTS dimensionality_reduction CASCADE;

CREATE SCHEMA dimensionality_reduction;

SET search_path TO dimensionality_reduction, public;

CREATE TYPE reduction_method AS ENUM (
    'pca',
    'tsne',
    'umap'
);

CREATE TYPE experiment_status AS ENUM (
    'created',
    'validated',
    'running',
    'completed',
    'failed'
);

CREATE TYPE observation_role AS ENUM (
    'training',
    'validation',
    'outlier',
    'reference'
);

CREATE TABLE dataset (
    dataset_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_name TEXT NOT NULL UNIQUE,
    feature_count INTEGER NOT NULL CHECK (feature_count >= 2),
    observation_count INTEGER NOT NULL CHECK (observation_count > 0),
    standardized BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE observation (
    observation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL
        REFERENCES dataset(dataset_id)
        ON DELETE CASCADE,
    external_key TEXT NOT NULL,
    role observation_role NOT NULL DEFAULT 'training',
    feature_values DOUBLE PRECISION[] NOT NULL,
    CHECK (cardinality(feature_values) >= 2),
    CHECK (
        NOT EXISTS (
            SELECT 1
            FROM unnest(feature_values) AS value
            WHERE value IS NULL OR value <> value
        )
    ),
    UNIQUE (dataset_id, external_key)
);

CREATE TABLE reduction_experiment (
    experiment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL
        REFERENCES dataset(dataset_id)
        ON DELETE RESTRICT,
    method reduction_method NOT NULL,
    output_dimensions INTEGER NOT NULL
        CHECK (output_dimensions BETWEEN 1 AND 3),
    perplexity DOUBLE PRECISION,
    neighbor_count INTEGER,
    learning_rate DOUBLE PRECISION NOT NULL
        CHECK (learning_rate > 0),
    min_dist DOUBLE PRECISION,
    iterations INTEGER NOT NULL
        CHECK (iterations > 0),
    random_seed INTEGER NOT NULL,
    status experiment_status NOT NULL DEFAULT 'created',
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    failure_reason TEXT,
    CHECK (
        method <> 'tsne'
        OR (
            perplexity IS NOT NULL
            AND perplexity > 1
            AND perplexity < 1000000
        )
    ),
    CHECK (
        method <> 'umap'
        OR (
            neighbor_count IS NOT NULL
            AND neighbor_count > 0
        )
    ),
    CHECK (
        method <> 'umap'
        OR (
            min_dist IS NOT NULL
            AND min_dist > 0
        )
    ),
    CHECK (
        completed_at IS NULL
        OR status = 'completed'
    )
);

CREATE TABLE embedding_point (
    embedding_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id BIGINT NOT NULL
        REFERENCES reduction_experiment(experiment_id)
        ON DELETE CASCADE,
    observation_id BIGINT NOT NULL
        REFERENCES observation(observation_id)
        ON DELETE CASCADE,
    coordinate_x DOUBLE PRECISION NOT NULL,
    coordinate_y DOUBLE PRECISION NOT NULL,
    coordinate_z DOUBLE PRECISION,
    UNIQUE (experiment_id, observation_id)
);

CREATE TABLE nearest_neighbor_edge (
    experiment_id BIGINT NOT NULL
        REFERENCES reduction_experiment(experiment_id)
        ON DELETE CASCADE,
    source_observation_id BIGINT NOT NULL
        REFERENCES observation(observation_id)
        ON DELETE CASCADE,
    target_observation_id BIGINT NOT NULL
        REFERENCES observation(observation_id)
        ON DELETE CASCADE,
    distance DOUBLE PRECISION NOT NULL
        CHECK (distance >= 0),
    affinity DOUBLE PRECISION NOT NULL
        CHECK (affinity BETWEEN 0 AND 1),
    PRIMARY KEY (
        experiment_id,
        source_observation_id,
        target_observation_id
    ),
    CHECK (
        source_observation_id <> target_observation_id
    )
);

CREATE TABLE status_check (
    status_check_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id BIGINT NOT NULL
        REFERENCES reduction_experiment(experiment_id)
        ON DELETE CASCADE,
    check_name TEXT NOT NULL,
    passed BOOLEAN NOT NULL,
    measured_value DOUBLE PRECISION,
    details TEXT,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (experiment_id, check_name)
);

CREATE INDEX idx_observation_dataset
    ON observation(dataset_id);

CREATE INDEX idx_experiment_dataset_method
    ON reduction_experiment(dataset_id, method);

CREATE INDEX idx_embedding_experiment
    ON embedding_point(experiment_id);

CREATE INDEX idx_neighbor_source
    ON nearest_neighbor_edge(experiment_id, source_observation_id);

CREATE INDEX idx_status_check_experiment
    ON status_check(experiment_id);

CREATE OR REPLACE FUNCTION validate_experiment_parameters()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    sample_count INTEGER;
BEGIN
    SELECT observation_count
    INTO sample_count
    FROM dataset
    WHERE dataset_id = NEW.dataset_id;

    IF sample_count IS NULL THEN
        RAISE EXCEPTION
            'Dataset % does not exist',
            NEW.dataset_id;
    END IF;

    IF NEW.method = 'tsne'
       AND NEW.perplexity >= sample_count THEN
        RAISE EXCEPTION
            't-SNE perplexity % must be smaller than sample count %',
            NEW.perplexity,
            sample_count;
    END IF;

    IF NEW.method = 'umap'
       AND NEW.neighbor_count >= sample_count THEN
        RAISE EXCEPTION
            'UMAP neighbor count % must be smaller than sample count %',
            NEW.neighbor_count,
            sample_count;
    END IF;

    IF NEW.status = 'running'
       AND NEW.started_at IS NULL THEN
        NEW.started_at := CURRENT_TIMESTAMP;
    END IF;

    IF NEW.status = 'completed'
       AND NEW.completed_at IS NULL THEN
        NEW.completed_at := CURRENT_TIMESTAMP;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_experiment
BEFORE INSERT OR UPDATE
ON reduction_experiment
FOR EACH ROW
EXECUTE FUNCTION validate_experiment_parameters();

CREATE OR REPLACE FUNCTION validate_embedding_coordinates()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NOT isfinite(NEW.coordinate_x)
       OR NOT isfinite(NEW.coordinate_y)
       OR (
           NEW.coordinate_z IS NOT NULL
           AND NOT isfinite(NEW.coordinate_z)
       ) THEN
        RAISE EXCEPTION
            'Embedding coordinates must be finite';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_embedding_coordinates
BEFORE INSERT OR UPDATE
ON embedding_point
FOR EACH ROW
EXECUTE FUNCTION validate_embedding_coordinates();

CREATE OR REPLACE FUNCTION prevent_completed_experiment_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    experiment_state experiment_status;
BEGIN
    SELECT status
    INTO experiment_state
    FROM reduction_experiment
    WHERE experiment_id = OLD.experiment_id;

    IF experiment_state = 'completed' THEN
        RAISE EXCEPTION
            'Completed experiment % cannot be modified',
            OLD.experiment_id;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_embedding_immutability
BEFORE UPDATE
ON embedding_point
FOR EACH ROW
EXECUTE FUNCTION prevent_completed_experiment_mutation();

INSERT INTO dataset (
    dataset_name,
    feature_count,
    observation_count,
    standardized
)
VALUES (
    'synthetic_manifold',
    4,
    12,
    TRUE
);

INSERT INTO observation (
    dataset_id,
    external_key,
    role,
    feature_values
)
SELECT
    d.dataset_id,
    'point-' || gs,
    CASE
        WHEN gs = 12 THEN 'outlier'::observation_role
        WHEN gs <= 8 THEN 'training'::observation_role
        ELSE 'validation'::observation_role
    END,
    ARRAY[
        sin(gs::double precision),
        cos(gs::double precision),
        gs::double precision / 10.0,
        sin(gs::double precision / 2.0)
    ]
FROM dataset AS d
CROSS JOIN generate_series(1, 12) AS gs
WHERE d.dataset_name = 'synthetic_manifold';

INSERT INTO reduction_experiment (
    dataset_id,
    method,
    output_dimensions,
    perplexity,
    neighbor_count,
    learning_rate,
    min_dist,
    iterations,
    random_seed,
    status
)
SELECT
    dataset_id,
    'tsne',
    2,
    5.0,
    NULL,
    50.0,
    NULL,
    250,
    42,
    'completed'
FROM dataset
WHERE dataset_name = 'synthetic_manifold';

INSERT INTO reduction_experiment (
    dataset_id,
    method,
    output_dimensions,
    perplexity,
    neighbor_count,
    learning_rate,
    min_dist,
    iterations,
    random_seed,
    status
)
SELECT
    dataset_id,
    'umap',
    2,
    NULL,
    5,
    50.0,
    0.1,
    250,
    42,
    'completed'
FROM dataset
WHERE dataset_name = 'synthetic_manifold';

INSERT INTO reduction_experiment (
    dataset_id,
    method,
    output_dimensions,
    perplexity,
    neighbor_count,
    learning_rate,
    min_dist,
    iterations,
    random_seed,
    status
)
SELECT
    dataset_id,
    'pca',
    2,
    NULL,
    NULL,
    1.0,
    NULL,
    1,
    42,
    'completed'
FROM dataset
WHERE dataset_name = 'synthetic_manifold';

INSERT INTO embedding_point (
    experiment_id,
    observation_id,
    coordinate_x,
    coordinate_y
)
SELECT
    e.experiment_id,
    o.observation_id,
    (
        o.feature_values[1] +
        o.feature_values[2]
    ) / 2.0,
    (
        o.feature_values[3] +
        o.feature_values[4]
    ) / 2.0
FROM reduction_experiment AS e
JOIN observation AS o
    ON o.dataset_id = e.dataset_id
WHERE e.method = 'pca';

INSERT INTO embedding_point (
    experiment_id,
    observation_id,
    coordinate_x,
    coordinate_y
)
SELECT
    e.experiment_id,
    o.observation_id,
    sin(o.observation_id::double precision),
    cos(o.observation_id::double precision)
FROM reduction_experiment AS e
JOIN observation AS o
    ON o.dataset_id = e.dataset_id
WHERE e.method = 'tsne';

INSERT INTO embedding_point (
    experiment_id,
    observation_id,
    coordinate_x,
    coordinate_y
)
SELECT
    e.experiment_id,
    o.observation_id,
    o.feature_values[1] * 0.8,
    o.feature_values[2] * 0.8
FROM reduction_experiment AS e
JOIN observation AS o
    ON o.dataset_id = e.dataset_id
WHERE e.method = 'umap';

INSERT INTO nearest_neighbor_edge (
    experiment_id,
    source_observation_id,
    target_observation_id,
    distance,
    affinity
)
SELECT
    e.experiment_id,
    source.observation_id,
    target.observation_id,
    sqrt(
        (
            source.feature_values[1] -
            target.feature_values[1]
        ) ^ 2 +
        (
            source.feature_values[2] -
            target.feature_values[2]
        ) ^ 2 +
        (
            source.feature_values[3] -
            target.feature_values[3]
        ) ^ 2 +
        (
            source.feature_values[4] -
            target.feature_values[4]
        ) ^ 2
    ),
    exp(
        -sqrt(
            (
                source.feature_values[1] -
                target.feature_values[1]
            ) ^ 2 +
            (
                source.feature_values[2] -
                target.feature_values[2]
            ) ^ 2 +
            (
                source.feature_values[3] -
                target.feature_values[3]
            ) ^ 2 +
            (
                source.feature_values[4] -
                target.feature_values[4]
            ) ^ 2
        )
    )
FROM reduction_experiment AS e
JOIN observation AS source
    ON source.dataset_id = e.dataset_id
JOIN observation AS target
    ON target.dataset_id = source.dataset_id
   AND target.observation_id <> source.observation_id
WHERE e.method = 'umap'
  AND target.observation_id = (
      SELECT o2.observation_id
      FROM observation AS o2
      WHERE o2.dataset_id = source.dataset_id
        AND o2.observation_id <> source.observation_id
      ORDER BY
          (
              o2.feature_values[1] -
              source.feature_values[1]
          ) ^ 2 +
          (
              o2.feature_values[2] -
              source.feature_values[2]
          ) ^ 2 +
          (
              o2.feature_values[3] -
              source.feature_values[3]
          ) ^ 2 +
          (
              o2.feature_values[4] -
              source.feature_values[4]
          ) ^ 2
      LIMIT 1
  );

INSERT INTO status_check (
    experiment_id,
    check_name,
    passed,
    measured_value,
    details
)
SELECT
    experiment_id,
    'embedding_row_count',
    COUNT(*) = (
        SELECT observation_count
        FROM dataset
        WHERE dataset_id = e.dataset_id
    ),
    COUNT(*),
    'Every source observation should have one embedding point'
FROM reduction_experiment AS e
LEFT JOIN embedding_point AS p
    ON p.experiment_id = e.experiment_id
GROUP BY e.experiment_id, e.dataset_id;

INSERT INTO status_check (
    experiment_id,
    check_name,
    passed,
    measured_value,
    details
)
SELECT
    experiment_id,
    'finite_coordinates',
    COUNT(*) > 0
        AND COUNT(*) FILTER (
            WHERE coordinate_x IS NULL
               OR coordinate_y IS NULL
        ) = 0,
    COUNT(*),
    'Embedding coordinates are populated'
FROM embedding_point
GROUP BY experiment_id;

CREATE VIEW experiment_summary AS
SELECT
    e.experiment_id,
    d.dataset_name,
    e.method,
    e.status,
    e.output_dimensions,
    e.iterations,
    e.perplexity,
    e.neighbor_count,
    e.learning_rate,
    e.min_dist,
    COUNT(p.embedding_id) AS embedding_points,
    COUNT(sc.status_check_id) AS status_checks,
    COUNT(sc.status_check_id)
        FILTER (WHERE sc.passed) AS passed_checks,
    e.started_at,
    e.completed_at
FROM reduction_experiment AS e
JOIN dataset AS d
    ON d.dataset_id = e.dataset_id
LEFT JOIN embedding_point AS p
    ON p.experiment_id = e.experiment_id
LEFT JOIN status_check AS sc
    ON sc.experiment_id = e.experiment_id
GROUP BY
    e.experiment_id,
    d.dataset_name,
    e.method,
    e.status,
    e.output_dimensions,
    e.iterations,
    e.perplexity,
    e.neighbor_count,
    e.learning_rate,
    e.min_dist,
    e.started_at,
    e.completed_at;

CREATE VIEW embedding_report AS
WITH source AS (
    SELECT
        o.observation_id,
        o.external_key,
        o.role,
        e.experiment_id,
        e.method,
        p.coordinate_x,
        p.coordinate_y
    FROM observation AS o
    JOIN embedding_point AS p
        ON p.observation_id = o.observation_id
    JOIN reduction_experiment AS e
        ON e.experiment_id = p.experiment_id
)
SELECT
    method,
    role,
    COUNT(*) AS observations,
    ROUND(AVG(coordinate_x)::numeric, 6) AS mean_x,
    ROUND(AVG(coordinate_y)::numeric, 6) AS mean_y,
    ROUND(
        MIN(coordinate_x)::numeric,
        6
    ) AS min_x,
    ROUND(
        MAX(coordinate_x)::numeric,
        6
    ) AS max_x,
    ROUND(
        MIN(coordinate_y)::numeric,
        6
    ) AS min_y,
    ROUND(
        MAX(coordinate_y)::numeric,
        6
    ) AS max_y
FROM source
GROUP BY method, role
ORDER BY method, role;

CREATE OR REPLACE FUNCTION evaluate_experiment(
    requested_experiment_id BIGINT
)
RETURNS TABLE (
    experiment_id BIGINT,
    method reduction_method,
    eligible BOOLEAN,
    reason TEXT
)
LANGUAGE plpgsql
AS $$
DECLARE
    experiment_record reduction_experiment%ROWTYPE;
    expected_count INTEGER;
    actual_count INTEGER;
    failed_checks INTEGER;
BEGIN
    SELECT *
    INTO experiment_record
    FROM reduction_experiment
    WHERE reduction_experiment.experiment_id =
        requested_experiment_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION
            'Experiment % does not exist',
            requested_experiment_id;
    END IF;

    SELECT observation_count
    INTO expected_count
    FROM dataset
    WHERE dataset_id = experiment_record.dataset_id;

    SELECT COUNT(*)
    INTO actual_count
    FROM embedding_point
    WHERE embedding_point.experiment_id =
        requested_experiment_id;

    SELECT COUNT(*)
    INTO failed_checks
    FROM status_check
    WHERE status_check.experiment_id =
        requested_experiment_id
      AND NOT passed;

    experiment_id := requested_experiment_id;
    method := experiment_record.method;

    IF experiment_record.status <> 'completed' THEN
        eligible := FALSE;
        reason := 'Experiment is not completed';
    ELSIF actual_count <> expected_count THEN
        eligible := FALSE;
        reason := 'Embedding cardinality does not match dataset';
    ELSIF failed_checks > 0 THEN
        eligible := FALSE;
        reason := 'One or more validation checks failed';
    ELSE
        eligible := TRUE;
        reason := 'Experiment passed structural validation';
    END IF;

    RETURN NEXT;
END;
$$;

-- Retrieval: compare experiment configurations.
SELECT *
FROM experiment_summary
ORDER BY method;

-- Validation: identify failed checks.
SELECT
    e.experiment_id,
    e.method,
    sc.check_name,
    sc.details
FROM reduction_experiment AS e
JOIN status_check AS sc
    ON sc.experiment_id = e.experiment_id
WHERE NOT sc.passed
ORDER BY e.experiment_id, sc.check_name;

-- Aggregation: compare embedding ranges by method.
SELECT *
FROM embedding_report;

-- Evaluation: determine whether each experiment is structurally valid.
SELECT *
FROM reduction_experiment
CROSS JOIN LATERAL evaluate_experiment(
    reduction_experiment.experiment_id
)
ORDER BY experiment_id;

-- Workflow-state reporting.
SELECT
    status,
    COUNT(*) AS experiment_count
FROM reduction_experiment
GROUP BY status
ORDER BY status;

-- Local-manifold reporting from the graph.
WITH ranked_edges AS (
    SELECT
        experiment_id,
        source_observation_id,
        target_observation_id,
        distance,
        affinity,
        ROW_NUMBER() OVER (
            PARTITION BY experiment_id, source_observation_id
            ORDER BY distance
        ) AS neighbor_rank
    FROM nearest_neighbor_edge
)
SELECT
    experiment_id,
    source_observation_id,
    target_observation_id,
    ROUND(distance::numeric, 6) AS distance,
    ROUND(affinity::numeric, 6) AS affinity
FROM ranked_edges
WHERE neighbor_rank <= 3
ORDER BY
    experiment_id,
    source_observation_id,
    neighbor_rank;

-- Transactional example: mark a running experiment as completed
-- only after its structural checks have been inserted.
BEGIN;

UPDATE reduction_experiment
SET status = 'completed',
    completed_at = CURRENT_TIMESTAMP
WHERE experiment_id = (
    SELECT experiment_id
    FROM reduction_experiment
    WHERE method = 'umap'
    ORDER BY experiment_id
    LIMIT 1
)
AND status IN ('created', 'running');

COMMIT;

-- The following statement is intentionally executable and demonstrates
-- a rejected parameter configuration through a savepoint.
BEGIN;

SAVEPOINT invalid_experiment;

DO $$
DECLARE
    dataset_key BIGINT;
BEGIN
    SELECT dataset_id
    INTO dataset_key
    FROM dataset
    WHERE dataset_name = 'synthetic_manifold';

    BEGIN
        INSERT INTO reduction_experiment (
            dataset_id,
            method,
            output_dimensions,
            perplexity,
            learning_rate,
            iterations,
            random_seed,
            status
        )
        VALUES (
            dataset_key,
            'tsne',
            2,
            50.0,
            50.0,
            100,
            7,
            'created'
        );
    EXCEPTION
        WHEN OTHERS THEN
            RAISE NOTICE
                'Rejected invalid t-SNE configuration: %',
                SQLERRM;
    END;
END;
$$;

ROLLBACK TO SAVEPOINT invalid_experiment;

COMMIT;
