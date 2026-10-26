DROP SCHEMA IF EXISTS advanced_clustering CASCADE;
CREATE SCHEMA advanced_clustering;

SET search_path = advanced_clustering, public;

-- The relational model keeps observations, algorithms, assignments, and
-- evaluation metrics separate so clustering runs can be reproduced.

CREATE TABLE datasets (
    dataset_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE observations (
    observation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    external_key TEXT NOT NULL,
    x DOUBLE PRECISION NOT NULL,
    y DOUBLE PRECISION NOT NULL,
    source_label TEXT,
    CONSTRAINT observations_coordinates_finite
        CHECK (
            isfinite(x)
            AND isfinite(y)
        ),
    CONSTRAINT observations_external_key_unique
        UNIQUE (dataset_id, external_key)
);

CREATE INDEX observations_dataset_idx
    ON observations(dataset_id);

CREATE TABLE clustering_runs (
    run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    algorithm TEXT NOT NULL,
    eps DOUBLE PRECISION,
    min_samples INTEGER,
    target_clusters INTEGER,
    linkage TEXT,
    status TEXT NOT NULL DEFAULT 'RUNNING',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT clustering_algorithm_check
        CHECK (
            algorithm IN ('HIERARCHICAL', 'DBSCAN')
        ),

    CONSTRAINT clustering_status_check
        CHECK (
            status IN ('RUNNING', 'COMPLETED', 'FAILED')
        ),

    CONSTRAINT clustering_eps_check
        CHECK (
            eps IS NULL OR eps > 0
        ),

    CONSTRAINT clustering_min_samples_check
        CHECK (
            min_samples IS NULL OR min_samples > 0
        ),

    CONSTRAINT clustering_target_check
        CHECK (
            target_clusters IS NULL OR target_clusters > 0
        ),

    CONSTRAINT clustering_linkage_check
        CHECK (
            linkage IS NULL
            OR linkage IN ('SINGLE', 'COMPLETE', 'AVERAGE')
        ),

    CONSTRAINT clustering_parameters_check
        CHECK (
            (
                algorithm = 'DBSCAN'
                AND eps IS NOT NULL
                AND min_samples IS NOT NULL
                AND target_clusters IS NULL
            )
            OR
            (
                algorithm = 'HIERARCHICAL'
                AND target_clusters IS NOT NULL
                AND linkage IS NOT NULL
                AND eps IS NULL
                AND min_samples IS NULL
            )
        )
);

CREATE INDEX clustering_runs_dataset_idx
    ON clustering_runs(dataset_id, created_at DESC);

CREATE TABLE clusters (
    cluster_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES clustering_runs(run_id) ON DELETE CASCADE,
    cluster_number INTEGER NOT NULL,
    is_noise BOOLEAN NOT NULL DEFAULT FALSE,

    CONSTRAINT clusters_number_check
        CHECK (cluster_number >= -1),

    CONSTRAINT clusters_unique_number
        UNIQUE (run_id, cluster_number)
);

CREATE TABLE cluster_memberships (
    run_id BIGINT NOT NULL REFERENCES clustering_runs(run_id) ON DELETE CASCADE,
    observation_id BIGINT NOT NULL REFERENCES observations(observation_id) ON DELETE CASCADE,
    cluster_id BIGINT NOT NULL REFERENCES clusters(cluster_id) ON DELETE CASCADE,
    distance_to_centroid DOUBLE PRECISION,

    PRIMARY KEY (run_id, observation_id),

    CONSTRAINT membership_distance_check
        CHECK (
            distance_to_centroid IS NULL
            OR distance_to_centroid >= 0
        )
);

CREATE INDEX cluster_memberships_cluster_idx
    ON cluster_memberships(cluster_id);

CREATE TABLE pairwise_distances (
    dataset_id BIGINT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    observation_a BIGINT NOT NULL REFERENCES observations(observation_id) ON DELETE CASCADE,
    observation_b BIGINT NOT NULL REFERENCES observations(observation_id) ON DELETE CASCADE,
    distance DOUBLE PRECISION NOT NULL,

    PRIMARY KEY (dataset_id, observation_a, observation_b),

    CONSTRAINT pairwise_order_check
        CHECK (observation_a < observation_b),

    CONSTRAINT pairwise_distance_check
        CHECK (distance >= 0)
);

CREATE INDEX pairwise_distance_lookup_idx
    ON pairwise_distances(dataset_id, distance);

CREATE TABLE clustering_metrics (
    run_id BIGINT PRIMARY KEY REFERENCES clustering_runs(run_id) ON DELETE CASCADE,
    silhouette_score DOUBLE PRECISION,
    cluster_count INTEGER,
    noise_count INTEGER,

    CONSTRAINT silhouette_range_check
        CHECK (
            silhouette_score IS NULL
            OR silhouette_score BETWEEN -1 AND 1
        ),

    CONSTRAINT metric_cluster_count_check
        CHECK (cluster_count IS NULL OR cluster_count >= 0),

    CONSTRAINT metric_noise_count_check
        CHECK (noise_count IS NULL OR noise_count >= 0)
);

INSERT INTO datasets (dataset_name)
VALUES ('service-observation-demo');

INSERT INTO observations
    (dataset_id, external_key, x, y, source_label)
SELECT
    d.dataset_id,
    v.external_key,
    v.x,
    v.y,
    v.source_label
FROM datasets d
CROSS JOIN (
    VALUES
        ('A1', 1.0::double precision, 1.0::double precision, 'group-a'),
        ('A2', 1.2::double precision, 0.9::double precision, 'group-a'),
        ('A3', 0.8::double precision, 1.1::double precision, 'group-a'),
        ('A4', 1.1::double precision, 1.3::double precision, 'group-a'),
        ('B1', 10.0::double precision, 10.0::double precision, 'group-b'),
        ('B2', 10.2::double precision, 10.1::double precision, 'group-b'),
        ('B3', 9.8::double precision, 10.3::double precision, 'group-b'),
        ('B4', 10.1::double precision, 9.7::double precision, 'group-b'),
        ('N1', 20.0::double precision, 2.0::double precision, 'outlier'),
        ('N2', 30.0::double precision, 30.0::double precision, 'outlier')
) AS v(external_key, x, y, source_label)
WHERE d.dataset_name = 'service-observation-demo';

-- Precomputing distances is useful when several clustering analyses operate
-- on the same small/medium dataset. For large datasets a spatial index is
-- preferable to materializing every pair.
INSERT INTO pairwise_distances (
    dataset_id,
    observation_a,
    observation_b,
    distance
)
SELECT
    a.dataset_id,
    a.observation_id,
    b.observation_id,
    sqrt(
        power(a.x - b.x, 2)
        + power(a.y - b.y, 2)
    )
FROM observations a
JOIN observations b
    ON a.dataset_id = b.dataset_id
   AND a.observation_id < b.observation_id;

-- A hierarchical run stores the requested number of final groups.
INSERT INTO clustering_runs (
    dataset_id,
    algorithm,
    target_clusters,
    linkage,
    status
)
SELECT
    dataset_id,
    'HIERARCHICAL',
    2,
    'AVERAGE',
    'COMPLETED'
FROM datasets
WHERE dataset_name = 'service-observation-demo';

-- A DBSCAN run stores density parameters separately from hierarchical
-- parameters because the algorithms have different semantics.
INSERT INTO clustering_runs (
    dataset_id,
    algorithm,
    eps,
    min_samples,
    status
)
SELECT
    dataset_id,
    'DBSCAN',
    1.0,
    3,
    'COMPLETED'
FROM datasets
WHERE dataset_name = 'service-observation-demo';

-- The following rows represent a realistic completed DBSCAN decision.
-- Cluster 0 and cluster 1 contain dense groups; cluster -1 represents noise.
INSERT INTO clusters (run_id, cluster_number, is_noise)
SELECT run_id, 0, FALSE
FROM clustering_runs
WHERE algorithm = 'DBSCAN'
UNION ALL
SELECT run_id, 1, FALSE
FROM clustering_runs
WHERE algorithm = 'DBSCAN'
UNION ALL
SELECT run_id, -1, TRUE
FROM clustering_runs
WHERE algorithm = 'DBSCAN';

-- Assign observations using their known geometric groups for the executable
-- demonstration. The database schema is capable of storing results produced
-- by an external clustering engine without mixing algorithm execution with
-- relational enforcement.
INSERT INTO cluster_memberships (
    run_id,
    observation_id,
    cluster_id
)
SELECT
    r.run_id,
    o.observation_id,
    c.cluster_id
FROM clustering_runs r
JOIN observations o
    ON o.dataset_id = r.dataset_id
JOIN clusters c
    ON c.run_id = r.run_id
WHERE r.algorithm = 'DBSCAN'
  AND (
      (o.source_label = 'group-a' AND c.cluster_number = 0)
      OR
      (o.source_label = 'group-b' AND c.cluster_number = 1)
      OR
      (o.source_label = 'outlier' AND c.cluster_number = -1)
  );

INSERT INTO clustering_metrics (
    run_id,
    cluster_count,
    noise_count
)
SELECT
    r.run_id,
    COUNT(*) FILTER (WHERE NOT c.is_noise),
    COUNT(*) FILTER (WHERE c.is_noise)
FROM clustering_runs r
JOIN clusters c ON c.run_id = r.run_id
WHERE r.algorithm = 'DBSCAN'
GROUP BY r.run_id;

-- Prevent membership records from referencing an observation belonging to
-- another dataset than the clustering run.
CREATE OR REPLACE FUNCTION enforce_membership_dataset()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    run_dataset BIGINT;
    observation_dataset BIGINT;
    cluster_run BIGINT;
BEGIN
    SELECT dataset_id
    INTO run_dataset
    FROM clustering_runs
    WHERE run_id = NEW.run_id;

    SELECT dataset_id
    INTO observation_dataset
    FROM observations
    WHERE observation_id = NEW.observation_id;

    SELECT run_id
    INTO cluster_run
    FROM clusters
    WHERE cluster_id = NEW.cluster_id;

    IF run_dataset IS NULL
       OR observation_dataset IS NULL
       OR run_dataset <> observation_dataset THEN
        RAISE EXCEPTION
            'Observation % does not belong to run % dataset',
            NEW.observation_id,
            NEW.run_id;
    END IF;

    IF cluster_run IS NULL OR cluster_run <> NEW.run_id THEN
        RAISE EXCEPTION
            'Cluster % does not belong to run %',
            NEW.cluster_id,
            NEW.run_id;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER membership_dataset_guard
BEFORE INSERT OR UPDATE ON cluster_memberships
FOR EACH ROW
EXECUTE FUNCTION enforce_membership_dataset();

-- Density-neighborhood query for DBSCAN parameter inspection.
WITH parameters AS (
    SELECT
        run_id,
        eps
    FROM clustering_runs
    WHERE algorithm = 'DBSCAN'
      AND status = 'COMPLETED'
)
SELECT
    p.run_id,
    o.observation_id,
    COUNT(n.observation_id) AS neighborhood_size,
    COUNT(n.observation_id) >= (
        SELECT min_samples
        FROM clustering_runs
        WHERE run_id = p.run_id
    ) AS is_core_candidate
FROM parameters p
JOIN observations o
    ON TRUE
LEFT JOIN observations n
    ON n.dataset_id = o.dataset_id
   AND sqrt(
       power(o.x - n.x, 2)
       + power(o.y - n.y, 2)
   ) <= p.eps
GROUP BY
    p.run_id,
    o.observation_id
ORDER BY
    p.run_id,
    o.observation_id;

-- Cluster-size reporting distinguishes density clusters from DBSCAN noise.
SELECT
    r.run_id,
    r.algorithm,
    c.cluster_number,
    c.is_noise,
    COUNT(cm.observation_id) AS member_count
FROM clustering_runs r
JOIN clusters c
    ON c.run_id = r.run_id
LEFT JOIN cluster_memberships cm
    ON cm.cluster_id = c.cluster_id
GROUP BY
    r.run_id,
    r.algorithm,
    c.cluster_number,
    c.is_noise
ORDER BY
    r.run_id,
    c.cluster_number;

-- Nearest-neighbor inspection is useful for diagnosing whether eps is too
-- small or too large for density-based clustering.
SELECT
    a.external_key AS observation_a,
    b.external_key AS observation_b,
    pd.distance
FROM pairwise_distances pd
JOIN observations a
    ON a.observation_id = pd.observation_a
JOIN observations b
    ON b.observation_id = pd.observation_b
WHERE pd.dataset_id = (
    SELECT dataset_id
    FROM datasets
    WHERE dataset_name = 'service-observation-demo'
)
ORDER BY pd.distance
LIMIT 15;

-- Transactional failure example. PostgreSQL rolls the entire transaction back
-- when the intentionally invalid membership is attempted.
BEGIN;

DO $$
DECLARE
    dbscan_run BIGINT;
    cluster_for_run BIGINT;
    invalid_observation BIGINT;
BEGIN
    SELECT run_id
    INTO dbscan_run
    FROM clustering_runs
    WHERE algorithm = 'DBSCAN'
    ORDER BY run_id DESC
    LIMIT 1;

    SELECT cluster_id
    INTO cluster_for_run
    FROM clusters
    WHERE run_id = dbscan_run
      AND cluster_number = 0;

    SELECT observation_id
    INTO invalid_observation
    FROM observations
    WHERE source_label = 'group-b'
    ORDER BY observation_id
    LIMIT 1;

    BEGIN
        INSERT INTO cluster_memberships (
            run_id,
            observation_id,
            cluster_id
        )
        VALUES (
            dbscan_run,
            invalid_observation,
            cluster_for_run
        );
    EXCEPTION
        WHEN OTHERS THEN
            RAISE NOTICE
                'Expected validation failure: %',
                SQLERRM;
    END;
END;
$$;

ROLLBACK;

-- Final operational report.
SELECT
    d.dataset_name,
    r.run_id,
    r.algorithm,
    r.eps,
    r.min_samples,
    r.target_clusters,
    r.linkage,
    r.status,
    m.cluster_count,
    m.noise_count
FROM datasets d
JOIN clustering_runs r
    ON r.dataset_id = d.dataset_id
LEFT JOIN clustering_metrics m
    ON m.run_id = r.run_id
ORDER BY r.run_id;
