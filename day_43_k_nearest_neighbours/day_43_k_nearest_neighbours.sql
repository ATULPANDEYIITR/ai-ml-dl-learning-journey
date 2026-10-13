-- k_nearest_neighbors.sql
-- PostgreSQL 15+
--
-- A relational k-NN workflow for predicting component quality status and
-- estimating inspection costs from historical measurements.
--
-- The SQL implementation uses explicit relational integrity, a standardized
-- feature table, distance calculations, window functions, neighborhood voting,
-- and regression aggregation.
--
-- The example uses three numeric features:
-- surface deviation, vibration score, and operating temperature.
-- Standardized measurements are stored separately from source measurements
-- so that the original observations remain auditable.

BEGIN;

DROP VIEW IF EXISTS knn_classification_results CASCADE;
DROP VIEW IF EXISTS knn_regression_results CASCADE;
DROP TABLE IF EXISTS prediction_requests CASCADE;
DROP TABLE IF EXISTS training_observations CASCADE;
DROP TABLE IF EXISTS standardized_observations CASCADE;
DROP TABLE IF EXISTS scaler_parameters CASCADE;
DROP TABLE IF EXISTS quality_components CASCADE;
DROP TYPE IF EXISTS quality_status CASCADE;

CREATE TYPE quality_status AS ENUM ('acceptable', 'review', 'reject');

CREATE TABLE quality_components (
    component_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    asset_code TEXT NOT NULL UNIQUE,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    surface_deviation NUMERIC(12, 4) NOT NULL,
    vibration_score NUMERIC(12, 4) NOT NULL,
    operating_temperature NUMERIC(12, 4) NOT NULL,
    quality_label quality_status NOT NULL,
    inspection_cost NUMERIC(12, 2) NOT NULL,
    CONSTRAINT surface_deviation_nonnegative CHECK (surface_deviation >= 0),
    CONSTRAINT vibration_score_nonnegative CHECK (vibration_score >= 0),
    CONSTRAINT inspection_cost_nonnegative CHECK (inspection_cost >= 0),
    CONSTRAINT measurements_finite_range CHECK (
        surface_deviation < 1000000
        AND vibration_score < 1000000
        AND operating_temperature > -273.15
        AND operating_temperature < 1000000
    )
);

CREATE TABLE scaler_parameters (
    feature_name TEXT PRIMARY KEY,
    training_mean DOUBLE PRECISION NOT NULL,
    training_scale DOUBLE PRECISION NOT NULL,
    fitted_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (training_scale > 0),
    CHECK (training_mean > '-Infinity'::DOUBLE PRECISION),
    CHECK (training_mean < 'Infinity'::DOUBLE PRECISION),
    CHECK (training_scale < 'Infinity'::DOUBLE PRECISION)
);

CREATE TABLE standardized_observations (
    component_id BIGINT PRIMARY KEY
        REFERENCES quality_components(component_id) ON DELETE CASCADE,
    z_surface DOUBLE PRECISION NOT NULL,
    z_vibration DOUBLE PRECISION NOT NULL,
    z_temperature DOUBLE PRECISION NOT NULL,
    scaler_version TEXT NOT NULL,
    CHECK (z_surface > '-Infinity'::DOUBLE PRECISION),
    CHECK (z_surface < 'Infinity'::DOUBLE PRECISION),
    CHECK (z_vibration > '-Infinity'::DOUBLE PRECISION),
    CHECK (z_vibration < 'Infinity'::DOUBLE PRECISION),
    CHECK (z_temperature > '-Infinity'::DOUBLE PRECISION),
    CHECK (z_temperature < 'Infinity'::DOUBLE PRECISION)
);

CREATE TABLE training_observations (
    component_id BIGINT PRIMARY KEY
        REFERENCES quality_components(component_id) ON DELETE CASCADE,
    training_partition TEXT NOT NULL DEFAULT 'train',
    included BOOLEAN NOT NULL DEFAULT TRUE,
    CHECK (training_partition IN ('train', 'validation', 'test'))
);

CREATE TABLE prediction_requests (
    request_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    request_code TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    surface_deviation NUMERIC(12, 4) NOT NULL,
    vibration_score NUMERIC(12, 4) NOT NULL,
    operating_temperature NUMERIC(12, 4) NOT NULL,
    z_surface DOUBLE PRECISION,
    z_vibration DOUBLE PRECISION,
    z_temperature DOUBLE PRECISION,
    neighbor_count INTEGER NOT NULL DEFAULT 3,
    weighting TEXT NOT NULL DEFAULT 'distance',
    CONSTRAINT positive_neighbor_count CHECK (neighbor_count > 0),
    CONSTRAINT supported_weighting CHECK (weighting IN ('uniform', 'distance')),
    CONSTRAINT request_surface_nonnegative CHECK (surface_deviation >= 0),
    CONSTRAINT request_vibration_nonnegative CHECK (vibration_score >= 0),
    CONSTRAINT request_temperature_valid CHECK (operating_temperature > -273.15)
);

CREATE INDEX quality_components_label_idx
    ON quality_components(quality_label);

CREATE INDEX training_partition_idx
    ON training_observations(training_partition, included);

CREATE INDEX prediction_requests_created_at_idx
    ON prediction_requests(created_at DESC);

INSERT INTO quality_components (
    asset_code,
    surface_deviation,
    vibration_score,
    operating_temperature,
    quality_label,
    inspection_cost
)
VALUES
    ('VALVE-201', 0.8, 1.2, 14.0, 'acceptable', 400.00),
    ('VALVE-202', 1.0, 1.5, 18.0, 'acceptable', 450.00),
    ('VALVE-203', 1.8, 2.7, 29.0, 'review', 1200.00),
    ('VALVE-204', 2.1, 3.1, 35.0, 'review', 1450.00),
    ('VALVE-205', 4.2, 5.7, 60.0, 'reject', 4000.00),
    ('VALVE-206', 4.8, 6.4, 72.0, 'reject', 5100.00),
    ('VALVE-207', 1.4, 2.0, 23.0, 'acceptable', 800.00),
    ('VALVE-208', 3.7, 4.9, 54.0, 'reject', 3500.00);

INSERT INTO training_observations (component_id, training_partition, included)
SELECT component_id, 'train', TRUE
FROM quality_components;

-- Calculate scaler parameters from training observations only.
-- Population standard deviation matches the convention used by the companion
-- implementations. A constant feature receives scale 1 to avoid division by zero.
INSERT INTO scaler_parameters (feature_name, training_mean, training_scale)
SELECT
    'surface_deviation',
    AVG(c.surface_deviation)::DOUBLE PRECISION,
    GREATEST(
        STDDEV_POP(c.surface_deviation)::DOUBLE PRECISION,
        1e-12
    )
FROM quality_components c
JOIN training_observations t USING (component_id)
WHERE t.included AND t.training_partition = 'train';

INSERT INTO scaler_parameters (feature_name, training_mean, training_scale)
SELECT
    'vibration_score',
    AVG(c.vibration_score)::DOUBLE PRECISION,
    GREATEST(
        STDDEV_POP(c.vibration_score)::DOUBLE PRECISION,
        1e-12
    )
FROM quality_components c
JOIN training_observations t USING (component_id)
WHERE t.included AND t.training_partition = 'train';

INSERT INTO scaler_parameters (feature_name, training_mean, training_scale)
SELECT
    'operating_temperature',
    AVG(c.operating_temperature)::DOUBLE PRECISION,
    GREATEST(
        STDDEV_POP(c.operating_temperature)::DOUBLE PRECISION,
        1e-12
    )
FROM quality_components c
JOIN training_observations t USING (component_id)
WHERE t.included AND t.training_partition = 'train';

INSERT INTO standardized_observations (
    component_id,
    z_surface,
    z_vibration,
    z_temperature,
    scaler_version
)
SELECT
    c.component_id,
    (c.surface_deviation::DOUBLE PRECISION - surface.training_mean)
        / surface.training_scale,
    (c.vibration_score::DOUBLE PRECISION - vibration.training_mean)
        / vibration.training_scale,
    (c.operating_temperature::DOUBLE PRECISION - temperature.training_mean)
        / temperature.training_scale,
    'training-scaler-v1'
FROM quality_components c
CROSS JOIN scaler_parameters surface
CROSS JOIN scaler_parameters vibration
CROSS JOIN scaler_parameters temperature
WHERE surface.feature_name = 'surface_deviation'
  AND vibration.feature_name = 'vibration_score'
  AND temperature.feature_name = 'operating_temperature';

INSERT INTO prediction_requests (
    request_code,
    surface_deviation,
    vibration_score,
    operating_temperature,
    neighbor_count,
    weighting
)
VALUES
    ('INCOMING-VALVE-301', 3.4, 4.6, 48.0, 3, 'distance'),
    ('INCOMING-VALVE-302', 1.1, 1.6, 19.0, 3, 'uniform');

UPDATE prediction_requests AS request
SET
    z_surface =
        (request.surface_deviation::DOUBLE PRECISION - surface.training_mean)
        / surface.training_scale,
    z_vibration =
        (request.vibration_score::DOUBLE PRECISION - vibration.training_mean)
        / vibration.training_scale,
    z_temperature =
        (request.operating_temperature::DOUBLE PRECISION - temperature.training_mean)
        / temperature.training_scale
FROM scaler_parameters AS surface,
     scaler_parameters AS vibration,
     scaler_parameters AS temperature
WHERE surface.feature_name = 'surface_deviation'
  AND vibration.feature_name = 'vibration_score'
  AND temperature.feature_name = 'operating_temperature';

-- Classification result view.
-- The row_number window function selects each request's k nearest training
-- records. Euclidean distance is calculated over standardized measurements.
CREATE VIEW knn_classification_results AS
WITH distances AS (
    SELECT
        request.request_id,
        request.request_code,
        request.neighbor_count,
        request.weighting,
        component.component_id,
        component.asset_code,
        component.quality_label,
        standardized.z_surface,
        standardized.z_vibration,
        standardized.z_temperature,
        SQRT(
            POWER(request.z_surface - standardized.z_surface, 2)
            + POWER(request.z_vibration - standardized.z_vibration, 2)
            + POWER(request.z_temperature - standardized.z_temperature, 2)
        ) AS distance
    FROM prediction_requests AS request
    CROSS JOIN quality_components AS component
    JOIN standardized_observations AS standardized USING (component_id)
    JOIN training_observations AS training USING (component_id)
    WHERE training.included
      AND training.training_partition = 'train'
),
ranked AS (
    SELECT
        distances.*,
        ROW_NUMBER() OVER (
            PARTITION BY request_id
            ORDER BY distance, component_id
        ) AS neighbor_rank
    FROM distances
),
selected_neighbors AS (
    SELECT *
    FROM ranked
    WHERE neighbor_rank <= neighbor_count
),
exact_match_flags AS (
    SELECT
        request_id,
        BOOL_OR(distance = 0) AS has_exact_match
    FROM selected_neighbors
    GROUP BY request_id
),
weighted_neighbors AS (
    SELECT
        neighbor.*,
        CASE
            WHEN neighbor.weighting = 'uniform' THEN 1.0
            WHEN flags.has_exact_match AND neighbor.distance = 0 THEN 1.0
            WHEN flags.has_exact_match THEN 0.0
            WHEN neighbor.distance = 0 THEN 1.0
            ELSE 1.0 / neighbor.distance
        END AS vote_weight
    FROM selected_neighbors AS neighbor
    JOIN exact_match_flags AS flags USING (request_id)
),
votes AS (
    SELECT
        request_id,
        request_code,
        quality_label,
        SUM(vote_weight) AS total_weight,
        SUM(distance) AS cumulative_distance
    FROM weighted_neighbors
    GROUP BY request_id, request_code, quality_label
),
ranked_votes AS (
    SELECT
        votes.*,
        SUM(total_weight) OVER (PARTITION BY request_id) AS all_weights,
        ROW_NUMBER() OVER (
            PARTITION BY request_id
            ORDER BY total_weight DESC, cumulative_distance ASC, quality_label
        ) AS class_rank
    FROM votes
)
SELECT
    request_id,
    request_code,
    quality_label AS predicted_status,
    total_weight / NULLIF(all_weights, 0) AS winning_vote_fraction
FROM ranked_votes
WHERE class_rank = 1;

-- Regression result view. Inverse-distance weighting gives closer historical
-- components greater influence. Exact matches suppress nonzero-distance records.
CREATE VIEW knn_regression_results AS
WITH distances AS (
    SELECT
        request.request_id,
        request.request_code,
        request.neighbor_count,
        request.weighting,
        component.component_id,
        component.inspection_cost,
        SQRT(
            POWER(request.z_surface - standardized.z_surface, 2)
            + POWER(request.z_vibration - standardized.z_vibration, 2)
            + POWER(request.z_temperature - standardized.z_temperature, 2)
        ) AS distance
    FROM prediction_requests AS request
    CROSS JOIN quality_components AS component
    JOIN standardized_observations AS standardized USING (component_id)
    JOIN training_observations AS training USING (component_id)
    WHERE training.included
      AND training.training_partition = 'train'
),
ranked AS (
    SELECT
        distances.*,
        ROW_NUMBER() OVER (
            PARTITION BY request_id
            ORDER BY distance, component_id
        ) AS neighbor_rank
    FROM distances
),
selected AS (
    SELECT *
    FROM ranked
    WHERE neighbor_rank <= neighbor_count
),
flags AS (
    SELECT request_id, BOOL_OR(distance = 0) AS has_exact_match
    FROM selected
    GROUP BY request_id
),
weighted AS (
    SELECT
        selected.*,
        CASE
            WHEN weighting = 'uniform' THEN 1.0
            WHEN flags.has_exact_match AND distance = 0 THEN 1.0
            WHEN flags.has_exact_match THEN 0.0
            WHEN distance = 0 THEN 1.0
            ELSE 1.0 / distance
        END AS weight
    FROM selected
    JOIN flags USING (request_id)
)
SELECT
    request_id,
    request_code,
    SUM(inspection_cost * weight) / NULLIF(SUM(weight), 0)
        AS estimated_inspection_cost
FROM weighted
GROUP BY request_id, request_code;

-- Inspect the neighbors chosen for each incoming component.
WITH distances AS (
    SELECT
        request.request_code,
        component.asset_code,
        component.quality_label,
        SQRT(
            POWER(request.z_surface - standardized.z_surface, 2)
            + POWER(request.z_vibration - standardized.z_vibration, 2)
            + POWER(request.z_temperature - standardized.z_temperature, 2)
        ) AS distance,
        ROW_NUMBER() OVER (
            PARTITION BY request.request_id
            ORDER BY
                SQRT(
                    POWER(request.z_surface - standardized.z_surface, 2)
                    + POWER(request.z_vibration - standardized.z_vibration, 2)
                    + POWER(request.z_temperature - standardized.z_temperature, 2)
                ),
                component.component_id
        ) AS neighbor_rank,
        request.neighbor_count
    FROM prediction_requests request
    CROSS JOIN quality_components component
    JOIN standardized_observations standardized USING (component_id)
    JOIN training_observations training USING (component_id)
    WHERE training.included
      AND training.training_partition = 'train'
)
SELECT request_code, asset_code, quality_label, distance, neighbor_rank
FROM distances
WHERE neighbor_rank <= neighbor_count
ORDER BY request_code, neighbor_rank;

SELECT * FROM knn_classification_results ORDER BY request_code;
SELECT * FROM knn_regression_results ORDER BY request_code;

-- Compare observed training labels with leave-one-out predictions.
-- Each historical component is excluded from its own neighbor set.
WITH leave_one_out_distances AS (
    SELECT
        query.component_id AS query_id,
        candidate.quality_label,
        SQRT(
            POWER(query_standardized.z_surface - candidate_standardized.z_surface, 2)
            + POWER(query_standardized.z_vibration - candidate_standardized.z_vibration, 2)
            + POWER(query_standardized.z_temperature - candidate_standardized.z_temperature, 2)
        ) AS distance,
        candidate.component_id AS candidate_id,
        ROW_NUMBER() OVER (
            PARTITION BY query.component_id
            ORDER BY
                SQRT(
                    POWER(query_standardized.z_surface - candidate_standardized.z_surface, 2)
                    + POWER(query_standardized.z_vibration - candidate_standardized.z_vibration, 2)
                    + POWER(query_standardized.z_temperature - candidate_standardized.z_temperature, 2)
                ),
                candidate.component_id
        ) AS neighbor_rank
    FROM quality_components query
    JOIN standardized_observations query_standardized
      ON query_standardized.component_id = query.component_id
    JOIN training_observations query_partition
      ON query_partition.component_id = query.component_id
    CROSS JOIN quality_components candidate
    JOIN standardized_observations candidate_standardized
      ON candidate_standardized.component_id = candidate.component_id
    JOIN training_observations candidate_partition
      ON candidate_partition.component_id = candidate.component_id
    WHERE query_partition.training_partition = 'train'
      AND query_partition.included
      AND candidate_partition.training_partition = 'train'
      AND candidate_partition.included
      AND query.component_id <> candidate.component_id
),
loo_votes AS (
    SELECT query_id, quality_label, COUNT(*) AS votes
    FROM leave_one_out_distances
    WHERE neighbor_rank <= 3
    GROUP BY query_id, quality_label
),
loo_predictions AS (
    SELECT DISTINCT ON (query_id)
        query_id,
        quality_label AS predicted_label
    FROM loo_votes
    ORDER BY query_id, votes DESC, quality_label
)
SELECT
    COUNT(*) AS evaluated_components,
    COUNT(*) FILTER (
        WHERE component.quality_label = prediction.predicted_label
    ) AS correct_predictions,
    ROUND(
        COUNT(*) FILTER (
            WHERE component.quality_label = prediction.predicted_label
        )::NUMERIC / NULLIF(COUNT(*), 0),
        3
    ) AS leave_one_out_accuracy
FROM quality_components component
JOIN loo_predictions prediction ON prediction.query_id = component.component_id;

-- The database constraints expose invalid values without silently changing them.
-- Uncomment the following statement to verify rejection of a negative cost:
-- INSERT INTO quality_components
--     (asset_code, surface_deviation, vibration_score, operating_temperature,
--      quality_label, inspection_cost)
-- VALUES ('INVALID-COST', 1.0, 1.0, 20.0, 'review', -5.00);

COMMIT;
