DROP SCHEMA IF EXISTS kernel_methods CASCADE;
CREATE SCHEMA kernel_methods;

SET search_path TO kernel_methods;

-- Kernel Methods are represented as a relational experiment platform.
-- Feature vectors are stored in PostgreSQL arrays so similarity operations
-- can be expressed directly in SQL while the catalog retains kernel
-- parameters and experiment metadata.

CREATE TYPE kernel_type AS ENUM (
    'linear',
    'polynomial',
    'rbf'
);

CREATE TYPE model_type AS ENUM (
    'similarity_classifier',
    'kernel_ridge'
);

CREATE TABLE kernel_definition (
    kernel_id BIGSERIAL PRIMARY KEY,
    kernel_name TEXT NOT NULL UNIQUE,
    kernel_type kernel_type NOT NULL,
    degree INTEGER,
    gamma DOUBLE PRECISION,
    coefficient DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT polynomial_degree_valid
        CHECK (
            kernel_type <> 'polynomial'
            OR (degree IS NOT NULL AND degree >= 1)
        ),

    CONSTRAINT gamma_valid
        CHECK (
            gamma IS NULL
            OR (gamma > 0 AND isfinite(gamma))
        ),

    CONSTRAINT coefficient_valid
        CHECK (
            coefficient IS NULL
            OR (coefficient >= 0 AND isfinite(coefficient))
        )
);

CREATE TABLE similarity_dataset (
    dataset_id BIGSERIAL PRIMARY KEY,
    dataset_name TEXT NOT NULL UNIQUE,
    feature_dimension INTEGER NOT NULL
        CHECK (feature_dimension > 0),
    description TEXT NOT NULL
);

CREATE TABLE sample (
    sample_id BIGSERIAL PRIMARY KEY,
    dataset_id BIGINT NOT NULL
        REFERENCES similarity_dataset(dataset_id)
        ON DELETE CASCADE,
    sample_name TEXT NOT NULL,
    features DOUBLE PRECISION[] NOT NULL,
    class_label INTEGER,
    regression_target DOUBLE PRECISION,

    CONSTRAINT feature_values_present
        CHECK (cardinality(features) > 0),

    CONSTRAINT regression_target_finite
        CHECK (
            regression_target IS NULL
            OR isfinite(regression_target)
        ),

    UNIQUE (dataset_id, sample_name)
);

CREATE INDEX sample_dataset_idx
    ON sample(dataset_id);

CREATE TABLE model (
    model_id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL UNIQUE,
    model_type model_type NOT NULL,
    dataset_id BIGINT NOT NULL
        REFERENCES similarity_dataset(dataset_id),
    kernel_id BIGINT NOT NULL
        REFERENCES kernel_definition(kernel_id),
    regularization DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT regularization_valid
        CHECK (
            regularization IS NULL
            OR (
                regularization > 0
                AND isfinite(regularization)
            )
        )
);

CREATE TABLE similarity_result (
    result_id BIGSERIAL PRIMARY KEY,
    model_id BIGINT NOT NULL
        REFERENCES model(model_id)
        ON DELETE CASCADE,
    query_name TEXT NOT NULL,
    query_features DOUBLE PRECISION[] NOT NULL,
    matched_sample_id BIGINT
        REFERENCES sample(sample_id),
    predicted_label INTEGER,
    similarity_score DOUBLE PRECISION NOT NULL
        CHECK (isfinite(similarity_score)),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE kernel_evaluation (
    evaluation_id BIGSERIAL PRIMARY KEY,
    kernel_id BIGINT NOT NULL
        REFERENCES kernel_definition(kernel_id)
        ON DELETE CASCADE,
    left_sample_id BIGINT NOT NULL
        REFERENCES sample(sample_id)
        ON DELETE CASCADE,
    right_sample_id BIGINT NOT NULL
        REFERENCES sample(sample_id)
        ON DELETE CASCADE,
    similarity DOUBLE PRECISION NOT NULL
        CHECK (
            isfinite(similarity)
        ),
    UNIQUE (
        kernel_id,
        left_sample_id,
        right_sample_id
    )
);

CREATE INDEX kernel_evaluation_left_idx
    ON kernel_evaluation(kernel_id, left_sample_id);

CREATE INDEX kernel_evaluation_right_idx
    ON kernel_evaluation(kernel_id, right_sample_id);

CREATE OR REPLACE FUNCTION validate_sample_dimension()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    expected_dimension INTEGER;
BEGIN
    SELECT feature_dimension
    INTO expected_dimension
    FROM similarity_dataset
    WHERE dataset_id = NEW.dataset_id;

    IF expected_dimension IS NULL THEN
        RAISE EXCEPTION
            'dataset % does not exist',
            NEW.dataset_id;
    END IF;

    IF cardinality(NEW.features) <> expected_dimension THEN
        RAISE EXCEPTION
            'sample % has dimension %, expected %',
            NEW.sample_name,
            cardinality(NEW.features),
            expected_dimension;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM unnest(NEW.features) AS value
        WHERE NOT isfinite(value)
    ) THEN
        RAISE EXCEPTION
            'sample % contains a non-finite feature',
            NEW.sample_name;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER sample_dimension_validation
BEFORE INSERT OR UPDATE OF dataset_id, features
ON sample
FOR EACH ROW
EXECUTE FUNCTION validate_sample_dimension();

CREATE OR REPLACE FUNCTION dot_product(
    left_features DOUBLE PRECISION[],
    right_features DOUBLE PRECISION[]
)
RETURNS DOUBLE PRECISION
LANGUAGE SQL
IMMUTABLE
STRICT
AS $$
    SELECT COALESCE(
        SUM(left_value * right_value),
        0
    )
    FROM unnest(left_features, right_features)
         AS pair(left_value, right_value);
$$;

CREATE OR REPLACE FUNCTION squared_distance(
    left_features DOUBLE PRECISION[],
    right_features DOUBLE PRECISION[]
)
RETURNS DOUBLE PRECISION
LANGUAGE SQL
IMMUTABLE
STRICT
AS $$
    SELECT COALESCE(
        SUM(
            (left_value - right_value)
            * (left_value - right_value)
        ),
        0
    )
    FROM unnest(left_features, right_features)
         AS pair(left_value, right_value);
$$;

CREATE OR REPLACE FUNCTION kernel_similarity(
    p_kernel_id BIGINT,
    left_features DOUBLE PRECISION[],
    right_features DOUBLE PRECISION[]
)
RETURNS DOUBLE PRECISION
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    definition kernel_definition%ROWTYPE;
    value DOUBLE PRECISION;
    effective_gamma DOUBLE PRECISION;
BEGIN
    SELECT *
    INTO definition
    FROM kernel_definition
    WHERE kernel_id = p_kernel_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION
            'kernel % does not exist',
            p_kernel_id;
    END IF;

    IF cardinality(left_features)
       <> cardinality(right_features) THEN
        RAISE EXCEPTION
            'kernel input dimensions differ';
    END IF;

    effective_gamma := COALESCE(
        definition.gamma,
        1.0 / cardinality(left_features)
    );

    CASE definition.kernel_type
        WHEN 'linear' THEN
            value := dot_product(
                left_features,
                right_features
            );

        WHEN 'polynomial' THEN
            value := power(
                effective_gamma *
                dot_product(
                    left_features,
                    right_features
                )
                + COALESCE(
                    definition.coefficient,
                    1.0
                ),
                definition.degree
            );

        WHEN 'rbf' THEN
            value := exp(
                -effective_gamma *
                squared_distance(
                    left_features,
                    right_features
                )
            );
    END CASE;

    RETURN value;
END;
$$;

CREATE OR REPLACE FUNCTION populate_kernel_evaluations(
    p_kernel_id BIGINT,
    p_dataset_id BIGINT
)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    inserted_count INTEGER;
BEGIN
    INSERT INTO kernel_evaluation (
        kernel_id,
        left_sample_id,
        right_sample_id,
        similarity
    )
    SELECT
        p_kernel_id,
        left_sample.sample_id,
        right_sample.sample_id,
        kernel_similarity(
            p_kernel_id,
            left_sample.features,
            right_sample.features
        )
    FROM sample AS left_sample
    CROSS JOIN sample AS right_sample
    WHERE left_sample.dataset_id = p_dataset_id
      AND right_sample.dataset_id = p_dataset_id
    ON CONFLICT (
        kernel_id,
        left_sample_id,
        right_sample_id
    )
    DO UPDATE SET
        similarity = EXCLUDED.similarity;

    GET DIAGNOSTICS inserted_count = ROW_COUNT;

    RETURN inserted_count;
END;
$$;

CREATE OR REPLACE VIEW kernel_similarity_report AS
SELECT
    ke.kernel_id,
    kd.kernel_name,
    kd.kernel_type,
    ke.left_sample_id,
    left_sample.sample_name AS left_sample,
    ke.right_sample_id,
    right_sample.sample_name AS right_sample,
    ke.similarity
FROM kernel_evaluation AS ke
JOIN kernel_definition AS kd
    ON kd.kernel_id = ke.kernel_id
JOIN sample AS left_sample
    ON left_sample.sample_id = ke.left_sample_id
JOIN sample AS right_sample
    ON right_sample.sample_id = ke.right_sample_id;

CREATE OR REPLACE VIEW dataset_statistics AS
SELECT
    d.dataset_id,
    d.dataset_name,
    d.feature_dimension,
    COUNT(s.sample_id) AS sample_count,
    COUNT(s.class_label) AS labeled_samples,
    COUNT(s.regression_target) AS regression_targets
FROM similarity_dataset AS d
LEFT JOIN sample AS s
    ON s.dataset_id = d.dataset_id
GROUP BY
    d.dataset_id,
    d.dataset_name,
    d.feature_dimension;

INSERT INTO kernel_definition (
    kernel_name,
    kernel_type,
    degree,
    gamma,
    coefficient
)
VALUES
    (
        'linear-default',
        'linear',
        NULL,
        NULL,
        NULL
    ),
    (
        'polynomial-degree-3',
        'polynomial',
        3,
        0.5,
        1.0
    ),
    (
        'rbf-xor',
        'rbf',
        NULL,
        2.0,
        NULL
    ),
    (
        'rbf-regression',
        'rbf',
        NULL,
        0.7,
        NULL
    );

INSERT INTO similarity_dataset (
    dataset_name,
    feature_dimension,
    description
)
VALUES (
    'xor_similarity',
    2,
    'Two-dimensional nonlinear classification data'
);

INSERT INTO similarity_dataset (
    dataset_name,
    feature_dimension,
    description
)
VALUES (
    'quadratic_regression',
    1,
    'One-dimensional samples with approximately quadratic targets'
);

INSERT INTO sample (
    dataset_id,
    sample_name,
    features,
    class_label
)
SELECT
    dataset_id,
    values.sample_name,
    values.features,
    values.class_label
FROM similarity_dataset AS d
CROSS JOIN (
    VALUES
        ('negative-negative', ARRAY[-1.0, -1.0]::DOUBLE PRECISION[], 0),
        ('negative-positive', ARRAY[-1.0,  1.0]::DOUBLE PRECISION[], 1),
        ('positive-negative', ARRAY[ 1.0, -1.0]::DOUBLE PRECISION[], 1),
        ('positive-positive', ARRAY[ 1.0,  1.0]::DOUBLE PRECISION[], 0),
        ('near-negative-negative', ARRAY[-0.8, -1.1]::DOUBLE PRECISION[], 0),
        ('near-negative-positive', ARRAY[-1.1,  0.8]::DOUBLE PRECISION[], 1),
        ('near-positive-negative', ARRAY[ 0.9, -1.2]::DOUBLE PRECISION[], 1),
        ('near-positive-positive', ARRAY[ 1.1,  1.0]::DOUBLE PRECISION[], 0)
) AS values(sample_name, features, class_label)
WHERE d.dataset_name = 'xor_similarity';

INSERT INTO sample (
    dataset_id,
    sample_name,
    features,
    regression_target
)
SELECT
    dataset_id,
    values.sample_name,
    values.features,
    values.regression_target
FROM similarity_dataset AS d
CROSS JOIN (
    VALUES
        ('x-minus-two', ARRAY[-2.0]::DOUBLE PRECISION[], 4.0),
        ('x-minus-one', ARRAY[-1.0]::DOUBLE PRECISION[], 1.0),
        ('x-zero', ARRAY[0.0]::DOUBLE PRECISION[], 0.0),
        ('x-one', ARRAY[1.0]::DOUBLE PRECISION[], 1.0),
        ('x-two', ARRAY[2.0]::DOUBLE PRECISION[], 4.0)
) AS values(sample_name, features, regression_target)
WHERE d.dataset_name = 'quadratic_regression';

INSERT INTO model (
    model_name,
    model_type,
    dataset_id,
    kernel_id,
    regularization
)
SELECT
    'xor-rbf-classifier',
    'similarity_classifier',
    d.dataset_id,
    k.kernel_id,
    NULL
FROM similarity_dataset AS d
JOIN kernel_definition AS k
    ON k.kernel_name = 'rbf-xor'
WHERE d.dataset_name = 'xor_similarity';

INSERT INTO model (
    model_name,
    model_type,
    dataset_id,
    kernel_id,
    regularization
)
SELECT
    'quadratic-rbf-regression',
    'kernel_ridge',
    d.dataset_id,
    k.kernel_id,
    0.05
FROM similarity_dataset AS d
JOIN kernel_definition AS k
    ON k.kernel_name = 'rbf-regression'
WHERE d.dataset_name = 'quadratic_regression';

BEGIN;

SELECT populate_kernel_evaluations(
    (
        SELECT kernel_id
        FROM kernel_definition
        WHERE kernel_name = 'rbf-xor'
    ),
    (
        SELECT dataset_id
        FROM similarity_dataset
        WHERE dataset_name = 'xor_similarity'
    )
);

SELECT populate_kernel_evaluations(
    (
        SELECT kernel_id
        FROM kernel_definition
        WHERE kernel_name = 'polynomial-degree-3'
    ),
    (
        SELECT dataset_id
        FROM similarity_dataset
        WHERE dataset_name = 'xor_similarity'
    )
);

COMMIT;

-- Inspect the similarity structure generated by the RBF kernel.
SELECT
    kernel_name,
    left_sample,
    right_sample,
    ROUND(similarity::numeric, 4) AS similarity
FROM kernel_similarity_report
WHERE kernel_name = 'rbf-xor'
ORDER BY
    left_sample,
    right_sample;

-- Find the most similar training sample to each query vector.
WITH queries AS (
    SELECT
        'query-negative-negative' AS query_name,
        ARRAY[-0.9, -0.9]::DOUBLE PRECISION[] AS features

    UNION ALL

    SELECT
        'query-negative-positive',
        ARRAY[-0.9, 0.9]::DOUBLE PRECISION[]

    UNION ALL

    SELECT
        'query-positive-negative',
        ARRAY[0.9, -0.9]::DOUBLE PRECISION[]

    UNION ALL

    SELECT
        'query-positive-positive',
        ARRAY[0.9, 0.9]::DOUBLE PRECISION[]
),
scored AS (
    SELECT
        q.query_name,
        s.sample_id,
        s.sample_name,
        s.class_label,
        kernel_similarity(
            (
                SELECT kernel_id
                FROM kernel_definition
                WHERE kernel_name = 'rbf-xor'
            ),
            q.features,
            s.features
        ) AS similarity
    FROM queries AS q
    CROSS JOIN sample AS s
    WHERE s.dataset_id = (
        SELECT dataset_id
        FROM similarity_dataset
        WHERE dataset_name = 'xor_similarity'
    )
)
SELECT DISTINCT ON (query_name)
    query_name,
    sample_name,
    class_label,
    ROUND(similarity::numeric, 6) AS similarity
FROM scored
ORDER BY
    query_name,
    similarity DESC;

-- Check that self-similarity is one for the RBF kernel.
SELECT
    sample_name,
    ROUND(
        kernel_similarity(
            (
                SELECT kernel_id
                FROM kernel_definition
                WHERE kernel_name = 'rbf-xor'
            ),
            features,
            features
        )::numeric,
        6
    ) AS self_similarity
FROM sample
WHERE dataset_id = (
    SELECT dataset_id
    FROM similarity_dataset
    WHERE dataset_name = 'xor_similarity'
)
ORDER BY sample_id;

-- The diagonal of a Gram matrix contains self-similarity values.
-- For RBF, these should equal one.
SELECT
    MIN(similarity) AS minimum_self_similarity,
    MAX(similarity) AS maximum_self_similarity
FROM kernel_evaluation
WHERE kernel_id = (
    SELECT kernel_id
    FROM kernel_definition
    WHERE kernel_name = 'rbf-xor'
)
AND left_sample_id = right_sample_id;

-- Demonstrate database-level dimension enforcement.
-- The statement is intentionally rolled back so the schema remains valid.
BEGIN;

DO $$
DECLARE
    dataset_identifier BIGINT;
BEGIN
    SELECT dataset_id
    INTO dataset_identifier
    FROM similarity_dataset
    WHERE dataset_name = 'xor_similarity';

    BEGIN
        INSERT INTO sample (
            dataset_id,
            sample_name,
            features,
            class_label
        )
        VALUES (
            dataset_identifier,
            'invalid-three-dimensional-sample',
            ARRAY[1.0, 2.0, 3.0],
            1
        );

        RAISE EXCEPTION
            'dimension validation unexpectedly accepted invalid data';

    EXCEPTION
        WHEN OTHERS THEN
            IF SQLERRM LIKE
                'dimension validation unexpectedly accepted invalid data%'
            THEN
                RAISE;
            END IF;

            RAISE NOTICE
                'Expected validation failure: %',
                SQLERRM;
    END;
END;
$$;

ROLLBACK;

-- Dataset-level reporting.
SELECT *
FROM dataset_statistics
ORDER BY dataset_id;

-- Kernel configuration report.
SELECT
    kernel_name,
    kernel_type,
    degree,
    gamma,
    coefficient
FROM kernel_definition
ORDER BY kernel_id;
