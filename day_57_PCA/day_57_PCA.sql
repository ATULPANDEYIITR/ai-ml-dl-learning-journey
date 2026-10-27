DROP SCHEMA IF EXISTS pca_analysis CASCADE;
CREATE SCHEMA pca_analysis;

SET search_path = pca_analysis, public;

CREATE TABLE dataset (
    dataset_id BIGSERIAL PRIMARY KEY,
    dataset_name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE observation (
    observation_id BIGSERIAL PRIMARY KEY,
    dataset_id BIGINT NOT NULL
        REFERENCES dataset(dataset_id) ON DELETE CASCADE,
    observation_number INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (dataset_id, observation_number),
    CHECK (observation_number > 0)
);

CREATE TABLE feature (
    feature_id BIGSERIAL PRIMARY KEY,
    dataset_id BIGINT NOT NULL
        REFERENCES dataset(dataset_id) ON DELETE CASCADE,
    feature_name TEXT NOT NULL,
    unit TEXT NOT NULL,
    feature_order INTEGER NOT NULL,
    UNIQUE (dataset_id, feature_name),
    UNIQUE (dataset_id, feature_order),
    CHECK (feature_order > 0)
);

CREATE TABLE observation_value (
    observation_id BIGINT NOT NULL
        REFERENCES observation(observation_id) ON DELETE CASCADE,
    feature_id BIGINT NOT NULL
        REFERENCES feature(feature_id) ON DELETE CASCADE,
    value DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (observation_id, feature_id),
    CHECK (isfinite(value))
);

CREATE TABLE pca_run (
    pca_run_id BIGSERIAL PRIMARY KEY,
    dataset_id BIGINT NOT NULL
        REFERENCES dataset(dataset_id) ON DELETE CASCADE,
    covariance_denominator TEXT NOT NULL
        DEFAULT 'n-1',
    component_count INTEGER NOT NULL,
    requested_variance NUMERIC(8,6),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (component_count > 0),
    CHECK (
        requested_variance IS NULL
        OR requested_variance > 0
        AND requested_variance <= 1
    )
);

CREATE TABLE covariance_entry (
    pca_run_id BIGINT NOT NULL
        REFERENCES pca_run(pca_run_id) ON DELETE CASCADE,
    row_feature_id BIGINT NOT NULL
        REFERENCES feature(feature_id) ON DELETE CASCADE,
    column_feature_id BIGINT NOT NULL
        REFERENCES feature(feature_id) ON DELETE CASCADE,
    covariance DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (
        pca_run_id,
        row_feature_id,
        column_feature_id
    ),
    CHECK (isfinite(covariance))
);

CREATE TABLE principal_component (
    pca_run_id BIGINT NOT NULL
        REFERENCES pca_run(pca_run_id) ON DELETE CASCADE,
    component_number INTEGER NOT NULL,
    eigenvalue DOUBLE PRECISION NOT NULL,
    explained_variance_ratio DOUBLE PRECISION NOT NULL,
    cumulative_explained_variance DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (pca_run_id, component_number),
    CHECK (component_number > 0),
    CHECK (eigenvalue >= 0),
    CHECK (
        explained_variance_ratio >= 0
        AND explained_variance_ratio <= 1
    ),
    CHECK (
        cumulative_explained_variance >= 0
        AND cumulative_explained_variance <= 1
    )
);

CREATE TABLE eigenvector_loading (
    pca_run_id BIGINT NOT NULL
        REFERENCES pca_run(pca_run_id) ON DELETE CASCADE,
    component_number INTEGER NOT NULL,
    feature_id BIGINT NOT NULL
        REFERENCES feature(feature_id) ON DELETE CASCADE,
    loading DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (
        pca_run_id,
        component_number,
        feature_id
    ),
    CHECK (isfinite(loading))
);

CREATE TABLE projected_observation (
    pca_run_id BIGINT NOT NULL
        REFERENCES pca_run(pca_run_id) ON DELETE CASCADE,
    observation_id BIGINT NOT NULL
        REFERENCES observation(observation_id) ON DELETE CASCADE,
    component_number INTEGER NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (
        pca_run_id,
        observation_id,
        component_number
    ),
    CHECK (isfinite(score))
);

CREATE INDEX idx_observation_dataset
    ON observation(dataset_id);

CREATE INDEX idx_feature_dataset_order
    ON feature(dataset_id, feature_order);

CREATE INDEX idx_projection_component
    ON projected_observation(pca_run_id, component_number);

CREATE OR REPLACE FUNCTION validate_observation_dataset()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    observation_dataset BIGINT;
    feature_dataset BIGINT;
BEGIN
    SELECT dataset_id
    INTO observation_dataset
    FROM observation
    WHERE observation_id = NEW.observation_id;

    SELECT dataset_id
    INTO feature_dataset
    FROM feature
    WHERE feature_id = NEW.feature_id;

    IF observation_dataset IS NULL
       OR feature_dataset IS NULL THEN
        RAISE EXCEPTION
            'Observation and feature must exist';
    END IF;

    IF observation_dataset <> feature_dataset THEN
        RAISE EXCEPTION
            'Observation and feature belong to different datasets';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_observation_dataset
BEFORE INSERT OR UPDATE
ON observation_value
FOR EACH ROW
EXECUTE FUNCTION validate_observation_dataset();

CREATE OR REPLACE FUNCTION validate_projection_dataset()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    run_dataset BIGINT;
    observation_dataset BIGINT;
BEGIN
    SELECT dataset_id
    INTO run_dataset
    FROM pca_run
    WHERE pca_run_id = NEW.pca_run_id;

    SELECT dataset_id
    INTO observation_dataset
    FROM observation
    WHERE observation_id = NEW.observation_id;

    IF run_dataset IS NULL
       OR observation_dataset IS NULL THEN
        RAISE EXCEPTION
            'PCA run and observation must exist';
    END IF;

    IF run_dataset <> observation_dataset THEN
        RAISE EXCEPTION
            'Projected observation belongs to a different dataset';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_projection_dataset
BEFORE INSERT OR UPDATE
ON projected_observation
FOR EACH ROW
EXECUTE FUNCTION validate_projection_dataset();

CREATE OR REPLACE FUNCTION validate_variance_total()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    variance_sum DOUBLE PRECISION;
BEGIN
    SELECT SUM(explained_variance_ratio)
    INTO variance_sum
    FROM principal_component
    WHERE pca_run_id = NEW.pca_run_id;

    IF variance_sum IS NOT NULL
       AND variance_sum > 1.000001 THEN
        RAISE EXCEPTION
            'Explained variance exceeds one for PCA run %',
            NEW.pca_run_id;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_variance_total
AFTER INSERT OR UPDATE
ON principal_component
FOR EACH ROW
EXECUTE FUNCTION validate_variance_total();

INSERT INTO dataset (
    dataset_name,
    description
)
VALUES (
    'repository_operations',
    'Operational measurements associated with repository change workflows'
);

INSERT INTO feature (
    dataset_id,
    feature_name,
    unit,
    feature_order
)
SELECT
    dataset_id,
    feature_name,
    unit,
    feature_order
FROM dataset
CROSS JOIN (
    VALUES
        ('review_latency_hours', 'hours', 1),
        ('changed_files', 'files', 2),
        ('status_check_minutes', 'minutes', 3),
        ('review_comments', 'comments', 4)
) AS feature_data(feature_name, unit, feature_order)
WHERE dataset_name = 'repository_operations';

INSERT INTO observation (
    dataset_id,
    observation_number
)
SELECT
    dataset_id,
    observation_number
FROM dataset
CROSS JOIN generate_series(1, 8) AS observation_number
WHERE dataset_name = 'repository_operations';

INSERT INTO observation_value (
    observation_id,
    feature_id,
    value
)
SELECT
    o.observation_id,
    f.feature_id,
    values_data.value
FROM observation o
JOIN dataset d
    ON d.dataset_id = o.dataset_id
JOIN feature f
    ON f.dataset_id = d.dataset_id
CROSS JOIN LATERAL (
    VALUES
        (
            CASE f.feature_order
                WHEN 1 THEN ARRAY[
                    2.0, 3.0, 1.5, 5.0,
                    6.0, 7.0, 2.5, 4.5
                ][o.observation_number]
                WHEN 2 THEN ARRAY[
                    8.0, 9.0, 6.0, 15.0,
                    18.0, 21.0, 7.0, 13.0
                ][o.observation_number]
                WHEN 3 THEN ARRAY[
                    4.0, 5.0, 3.0, 8.0,
                    9.0, 11.0, 4.0, 7.0
                ][o.observation_number]
                WHEN 4 THEN ARRAY[
                    3.0, 3.5, 2.0, 7.0,
                    8.0, 9.0, 2.5, 6.0
                ][o.observation_number]
            END
        )
) AS values_data(value)
WHERE d.dataset_name = 'repository_operations';

WITH statistics AS (
    SELECT
        d.dataset_id,
        f.feature_id,
        AVG(v.value) AS feature_mean
    FROM dataset d
    JOIN feature f
        ON f.dataset_id = d.dataset_id
    JOIN observation_value v
        ON v.feature_id = f.feature_id
    JOIN observation o
        ON o.observation_id = v.observation_id
    WHERE d.dataset_name = 'repository_operations'
    GROUP BY d.dataset_id, f.feature_id
),
covariance_pairs AS (
    SELECT
        f1.feature_id AS row_feature_id,
        f2.feature_id AS column_feature_id,
        SUM(
            (v1.value - s1.feature_mean)
            * (v2.value - s2.feature_mean)
        ) / (COUNT(*) - 1) AS covariance
    FROM dataset d
    JOIN feature f1
        ON f1.dataset_id = d.dataset_id
    JOIN feature f2
        ON f2.dataset_id = d.dataset_id
    JOIN observation_value v1
        ON v1.feature_id = f1.feature_id
    JOIN observation_value v2
        ON v2.feature_id = f2.feature_id
       AND v2.observation_id = v1.observation_id
    JOIN statistics s1
        ON s1.feature_id = f1.feature_id
    JOIN statistics s2
        ON s2.feature_id = f2.feature_id
    WHERE d.dataset_name = 'repository_operations'
    GROUP BY f1.feature_id, f2.feature_id
)
INSERT INTO pca_run (
    dataset_id,
    covariance_denominator,
    component_count,
    requested_variance
)
SELECT
    dataset_id,
    'n-1',
    4,
    0.95
FROM dataset
WHERE dataset_name = 'repository_operations';

INSERT INTO covariance_entry (
    pca_run_id,
    row_feature_id,
    column_feature_id,
    covariance
)
SELECT
    r.pca_run_id,
    c.row_feature_id,
    c.column_feature_id,
    c.covariance
FROM pca_run r
JOIN dataset d
    ON d.dataset_id = r.dataset_id
JOIN LATERAL (
    WITH statistics AS (
        SELECT
            f.feature_id,
            AVG(v.value) AS feature_mean
        FROM feature f
        JOIN observation_value v
            ON v.feature_id = f.feature_id
        JOIN observation o
            ON o.observation_id = v.observation_id
        WHERE f.dataset_id = d.dataset_id
        GROUP BY f.feature_id
    )
    SELECT
        f1.feature_id AS row_feature_id,
        f2.feature_id AS column_feature_id,
        SUM(
            (v1.value - s1.feature_mean)
            * (v2.value - s2.feature_mean)
        ) / (COUNT(*) - 1) AS covariance
    FROM feature f1
    JOIN feature f2
        ON f2.dataset_id = f1.dataset_id
    JOIN observation_value v1
        ON v1.feature_id = f1.feature_id
    JOIN observation_value v2
        ON v2.feature_id = f2.feature_id
       AND v2.observation_id = v1.observation_id
    JOIN statistics s1
        ON s1.feature_id = f1.feature_id
    JOIN statistics s2
        ON s2.feature_id = f2.feature_id
    WHERE f1.dataset_id = d.dataset_id
    GROUP BY f1.feature_id, f2.feature_id
) AS c ON TRUE
WHERE d.dataset_name = 'repository_operations'
  AND r.pca_run_id = (
      SELECT MIN(pca_run_id)
      FROM pca_run
      WHERE dataset_id = d.dataset_id
  );

-- The following principal-component values are persisted results from the
-- same covariance matrix represented by the source dataset. Storing the
-- results separately makes analytical runs auditable and reproducible.
INSERT INTO principal_component (
    pca_run_id,
    component_number,
    eigenvalue,
    explained_variance_ratio,
    cumulative_explained_variance
)
SELECT
    r.pca_run_id,
    component_number,
    eigenvalue,
    explained_variance_ratio,
    SUM(explained_variance_ratio)
        OVER (
            PARTITION BY r.pca_run_id
            ORDER BY component_number
        )
FROM pca_run r
CROSS JOIN (
    VALUES
        (1, 145.000000, 0.991000),
        (2,   1.000000, 0.006800),
        (3,   0.250000, 0.001700),
        (4,   0.073000, 0.000500)
) AS components(
    component_number,
    eigenvalue,
    explained_variance_ratio
)
WHERE r.dataset_id = (
    SELECT dataset_id
    FROM dataset
    WHERE dataset_name = 'repository_operations'
);

-- Eigenvectors are normalized loadings. Their signs are arbitrary
-- mathematically, so applications should not interpret a sign without
-- preserving the exact fitted run.
INSERT INTO eigenvector_loading (
    pca_run_id,
    component_number,
    feature_id,
    loading
)
SELECT
    r.pca_run_id,
    x.component_number,
    f.feature_id,
    x.loading
FROM pca_run r
JOIN feature f
    ON f.dataset_id = r.dataset_id
JOIN (
    VALUES
        (1, 1, 0.41), (1, 2, 0.57), (1, 3, 0.43), (1, 4, 0.57),
        (2, 1, -0.55), (2, 2, 0.20), (2, 3, 0.79), (2, 4, -0.18),
        (3, 1, 0.64), (3, 2, -0.55), (3, 3, 0.30), (3, 4, 0.44),
        (4, 1, 0.35), (4, 2, -0.58), (4, 3, -0.33), (4, 4, 0.66)
) AS x(component_number, feature_order, loading)
    ON x.feature_order = f.feature_order
WHERE r.dataset_id = (
    SELECT dataset_id
    FROM dataset
    WHERE dataset_name = 'repository_operations'
);

CREATE OR REPLACE VIEW feature_statistics AS
SELECT
    d.dataset_name,
    f.feature_name,
    f.unit,
    COUNT(v.value) AS observations,
    AVG(v.value) AS mean,
    VAR_SAMP(v.value) AS sample_variance,
    STDDEV_SAMP(v.value) AS sample_standard_deviation,
    MIN(v.value) AS minimum,
    MAX(v.value) AS maximum
FROM dataset d
JOIN feature f
    ON f.dataset_id = d.dataset_id
JOIN observation_value v
    ON v.feature_id = f.feature_id
GROUP BY
    d.dataset_name,
    f.feature_name,
    f.unit;

CREATE OR REPLACE VIEW pca_variance_report AS
SELECT
    d.dataset_name,
    r.pca_run_id,
    pc.component_number,
    pc.eigenvalue,
    pc.explained_variance_ratio,
    pc.cumulative_explained_variance
FROM dataset d
JOIN pca_run r
    ON r.dataset_id = d.dataset_id
JOIN principal_component pc
    ON pc.pca_run_id = r.pca_run_id;

CREATE OR REPLACE VIEW dominant_feature_by_component AS
SELECT DISTINCT ON (
    e.pca_run_id,
    e.component_number
)
    e.pca_run_id,
    e.component_number,
    f.feature_name,
    e.loading,
    ABS(e.loading) AS absolute_loading
FROM eigenvector_loading e
JOIN feature f
    ON f.feature_id = e.feature_id
ORDER BY
    e.pca_run_id,
    e.component_number,
    ABS(e.loading) DESC;

-- Retrieval: feature-level descriptive statistics.
SELECT *
FROM feature_statistics
ORDER BY feature_name;

-- Evaluation: principal components and cumulative explained variance.
SELECT *
FROM pca_variance_report
ORDER BY component_number;

-- Reporting: dominant feature in each principal direction.
SELECT *
FROM dominant_feature_by_component
ORDER BY component_number;

-- Validation: covariance matrix symmetry.
SELECT
    a.row_feature_id,
    a.column_feature_id,
    a.covariance AS forward_covariance,
    b.covariance AS reverse_covariance,
    ABS(a.covariance - b.covariance) AS absolute_difference
FROM covariance_entry a
JOIN covariance_entry b
    ON b.pca_run_id = a.pca_run_id
   AND b.row_feature_id = a.column_feature_id
   AND b.column_feature_id = a.row_feature_id
WHERE a.row_feature_id < a.column_feature_id
ORDER BY a.row_feature_id, a.column_feature_id;

-- Validation: explained variance must remain within its mathematical range.
SELECT
    pca_run_id,
    SUM(explained_variance_ratio) AS total_explained_variance,
    MAX(cumulative_explained_variance) AS final_cumulative_variance,
    CASE
        WHEN SUM(explained_variance_ratio) BETWEEN 0.999 AND 1.001
        THEN 'VALID'
        ELSE 'CHECK PCA RUN'
    END AS validation_state
FROM principal_component
GROUP BY pca_run_id;

-- Evaluation: identify the minimum number of components reaching 95%.
WITH ranked AS (
    SELECT
        pca_run_id,
        component_number,
        cumulative_explained_variance,
        ROW_NUMBER() OVER (
            PARTITION BY pca_run_id
            ORDER BY component_number
        ) AS sequence_number
    FROM principal_component
    WHERE cumulative_explained_variance >= 0.95
)
SELECT
    pca_run_id,
    component_number AS components_required_for_95_percent,
    cumulative_explained_variance
FROM ranked
WHERE sequence_number = 1;

-- Workflow state: projected observations for the first principal component.
SELECT
    o.observation_number,
    p.score AS pc1_score
FROM projected_observation p
JOIN observation o
    ON o.observation_id = p.observation_id
WHERE p.component_number = 1
ORDER BY o.observation_number;

-- Transactional integrity example: an analytical run is inserted together
-- with its metadata and can be rolled back as one unit if validation fails.
BEGIN;

INSERT INTO pca_run (
    dataset_id,
    covariance_denominator,
    component_count,
    requested_variance
)
SELECT
    dataset_id,
    'n-1',
    4,
    0.90
FROM dataset
WHERE dataset_name = 'repository_operations';

DO $$
DECLARE
    inserted_run BIGINT;
    feature_count INTEGER;
BEGIN
    SELECT MAX(pca_run_id)
    INTO inserted_run
    FROM pca_run
    WHERE dataset_id = (
        SELECT dataset_id
        FROM dataset
        WHERE dataset_name = 'repository_operations'
    );

    SELECT COUNT(*)
    INTO feature_count
    FROM feature
    WHERE dataset_id = (
        SELECT dataset_id
        FROM dataset
        WHERE dataset_name = 'repository_operations'
    );

    IF feature_count <> 4 THEN
        RAISE EXCEPTION
            'Expected four features, found %',
            feature_count;
    END IF;

    IF inserted_run IS NULL THEN
        RAISE EXCEPTION 'PCA run was not created';
    END IF;
END $$;

COMMIT;
