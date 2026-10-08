DROP SCHEMA IF EXISTS regularization_demo CASCADE;
CREATE SCHEMA regularization_demo;

SET search_path TO regularization_demo;

CREATE TABLE experiments (
    experiment_id BIGSERIAL PRIMARY KEY,
    experiment_name TEXT NOT NULL UNIQUE,
    target_variable TEXT NOT NULL,
    observations INTEGER NOT NULL CHECK (observations > 0),
    feature_count INTEGER NOT NULL CHECK (feature_count > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TYPE model_family AS ENUM (
    'OLS',
    'RIDGE',
    'LASSO',
    'ELASTIC_NET'
);

CREATE TABLE models (
    model_id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    family model_family NOT NULL,
    alpha NUMERIC(12, 6) NOT NULL CHECK (alpha >= 0),
    l1_ratio NUMERIC(8, 6),
    standardized_features BOOLEAN NOT NULL DEFAULT TRUE,
    training_status TEXT NOT NULL
        CHECK (
            training_status IN (
                'TRAINED',
                'FAILED',
                'VALIDATION_PENDING'
            )
        ),
    UNIQUE (experiment_id, model_name),
    CHECK (
        (family = 'ELASTIC_NET'
            AND l1_ratio IS NOT NULL
            AND l1_ratio BETWEEN 0 AND 1)
        OR
        (family <> 'ELASTIC_NET'
            AND l1_ratio IS NULL)
    )
);

CREATE TABLE features (
    feature_id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,
    feature_name TEXT NOT NULL,
    feature_role TEXT NOT NULL
        CHECK (
            feature_role IN (
                'SIGNAL',
                'CORRELATED_SIGNAL',
                'NOISE',
                'BINARY'
            )
        ),
    source_unit TEXT NOT NULL,
    UNIQUE (experiment_id, feature_name)
);

CREATE TABLE coefficients (
    model_id BIGINT NOT NULL
        REFERENCES models(model_id)
        ON DELETE CASCADE,
    feature_id BIGINT NOT NULL
        REFERENCES features(feature_id)
        ON DELETE CASCADE,
    coefficient_value NUMERIC(18, 8) NOT NULL,
    absolute_value NUMERIC(18, 8)
        GENERATED ALWAYS AS (
            ABS(coefficient_value)
        ) STORED,
    is_zero BOOLEAN
        GENERATED ALWAYS AS (
            ABS(coefficient_value) < 0.00000001
        ) STORED,
    PRIMARY KEY (model_id, feature_id)
);

CREATE INDEX idx_coefficients_feature
    ON coefficients(feature_id);

CREATE INDEX idx_coefficients_nonzero
    ON coefficients(model_id, is_zero);

CREATE TABLE observations (
    observation_id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,
    split TEXT NOT NULL
        CHECK (split IN ('TRAIN', 'VALIDATION', 'TEST')),
    temperature NUMERIC(12, 4) NOT NULL,
    humidity NUMERIC(12, 4) NOT NULL,
    pressure NUMERIC(12, 4) NOT NULL,
    advertising_spend NUMERIC(12, 4) NOT NULL
        CHECK (advertising_spend >= 0),
    discount_rate NUMERIC(12, 4) NOT NULL
        CHECK (discount_rate BETWEEN 0 AND 100),
    holiday_flag BOOLEAN NOT NULL,
    competitor_price NUMERIC(12, 4) NOT NULL
        CHECK (competitor_price > 0),
    store_traffic INTEGER NOT NULL
        CHECK (store_traffic >= 0),
    target_value NUMERIC(14, 4) NOT NULL
);

CREATE INDEX idx_observations_experiment_split
    ON observations(experiment_id, split);

CREATE TABLE model_metrics (
    model_id BIGINT PRIMARY KEY
        REFERENCES models(model_id)
        ON DELETE CASCADE,
    train_mse NUMERIC(18, 8) CHECK (train_mse >= 0),
    validation_mse NUMERIC(18, 8) CHECK (validation_mse >= 0),
    test_mse NUMERIC(18, 8) CHECK (test_mse >= 0),
    train_r2 NUMERIC(18, 8),
    validation_r2 NUMERIC(18, 8),
    test_r2 NUMERIC(18, 8),
    fitted_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE regularization_runs (
    run_id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,
    family model_family NOT NULL,
    alpha NUMERIC(12, 6) NOT NULL CHECK (alpha >= 0),
    l1_ratio NUMERIC(8, 6),
    validation_mse NUMERIC(18, 8) NOT NULL
        CHECK (validation_mse >= 0),
    selected BOOLEAN NOT NULL DEFAULT FALSE,
    CHECK (
        (family = 'ELASTIC_NET'
            AND l1_ratio IS NOT NULL
            AND l1_ratio BETWEEN 0 AND 1)
        OR
        (family <> 'ELASTIC_NET'
            AND l1_ratio IS NULL)
    )
);

CREATE INDEX idx_regularization_runs_selection
    ON regularization_runs(
        experiment_id,
        family,
        selected
    );

INSERT INTO experiments (
    experiment_name,
    target_variable,
    observations,
    feature_count
)
VALUES (
    'Retail Demand Regularization Study',
    'daily_demand',
    180,
    12
);

INSERT INTO features (
    experiment_id,
    feature_name,
    feature_role,
    source_unit
)
VALUES
    (
        1,
        'temperature',
        'SIGNAL',
        'celsius'
    ),
    (
        1,
        'humidity',
        'SIGNAL',
        'percent'
    ),
    (
        1,
        'pressure',
        'CORRELATED_SIGNAL',
        'hPa'
    ),
    (
        1,
        'advertising_spend',
        'SIGNAL',
        'currency'
    ),
    (
        1,
        'discount_rate',
        'SIGNAL',
        'percent'
    ),
    (
        1,
        'holiday_flag',
        'BINARY',
        'boolean'
    ),
    (
        1,
        'competitor_price',
        'SIGNAL',
        'currency'
    ),
    (
        1,
        'store_traffic',
        'SIGNAL',
        'customers'
    ),
    (
        1,
        'noise_feature_a',
        'NOISE',
        'standardized'
    ),
    (
        1,
        'noise_feature_b',
        'NOISE',
        'standardized'
    ),
    (
        1,
        'noise_feature_c',
        'NOISE',
        'standardized'
    ),
    (
        1,
        'noise_feature_d',
        'NOISE',
        'standardized'
    );

INSERT INTO models (
    experiment_id,
    model_name,
    family,
    alpha,
    l1_ratio,
    standardized_features,
    training_status
)
VALUES
    (
        1,
        'OLS Baseline',
        'OLS',
        0,
        NULL,
        TRUE,
        'TRAINED'
    ),
    (
        1,
        'Ridge Candidate',
        'RIDGE',
        0.8,
        NULL,
        TRUE,
        'TRAINED'
    ),
    (
        1,
        'Lasso Candidate',
        'LASSO',
        0.08,
        NULL,
        TRUE,
        'TRAINED'
    ),
    (
        1,
        'Elastic Net Candidate',
        'ELASTIC_NET',
        0.08,
        0.5,
        TRUE,
        'TRAINED'
    );

INSERT INTO observations (
    experiment_id,
    split,
    temperature,
    humidity,
    pressure,
    advertising_spend,
    discount_rate,
    holiday_flag,
    competitor_price,
    store_traffic,
    target_value
)
VALUES
    (
        1,
        'TRAIN',
        21.2,
        55.0,
        1038.2,
        70.0,
        10.0,
        FALSE,
        49.0,
        470,
        111.4
    ),
    (
        1,
        'TRAIN',
        25.0,
        63.0,
        1044.7,
        90.0,
        15.0,
        TRUE,
        52.0,
        620,
        161.3
    ),
    (
        1,
        'TRAIN',
        18.5,
        72.0,
        1033.4,
        20.0,
        5.0,
        FALSE,
        47.0,
        390,
        88.7
    ),
    (
        1,
        'TRAIN',
        27.1,
        50.0,
        1048.4,
        65.0,
        20.0,
        FALSE,
        55.0,
        680,
        145.9
    ),
    (
        1,
        'VALIDATION',
        23.4,
        61.0,
        1041.8,
        75.0,
        12.0,
        FALSE,
        51.0,
        530,
        128.5
    ),
    (
        1,
        'VALIDATION',
        19.8,
        58.0,
        1035.5,
        35.0,
        7.0,
        TRUE,
        48.0,
        450,
        116.9
    ),
    (
        1,
        'TEST',
        24.6,
        57.0,
        1043.0,
        82.0,
        18.0,
        FALSE,
        53.0,
        590,
        143.7
    ),
    (
        1,
        'TEST',
        17.9,
        68.0,
        1031.9,
        25.0,
        3.0,
        FALSE,
        46.0,
        360,
        83.2
    );

INSERT INTO coefficients (
    model_id,
    feature_id,
    coefficient_value
)
VALUES
    (1, 1, 0.51),
    (1, 2, -0.18),
    (1, 3, 0.09),
    (1, 4, 0.31),
    (1, 5, 0.21),
    (1, 6, 0.08),
    (1, 7, -0.13),
    (1, 8, 0.24),
    (1, 9, 0.16),
    (1, 10, -0.11),
    (1, 11, 0.07),
    (1, 12, -0.05),
    (2, 1, 0.45),
    (2, 2, -0.16),
    (2, 3, 0.08),
    (2, 4, 0.29),
    (2, 5, 0.19),
    (2, 6, 0.07),
    (2, 7, -0.12),
    (2, 8, 0.21),
    (2, 9, 0.08),
    (2, 10, -0.05),
    (2, 11, 0.03),
    (2, 12, -0.02),
    (3, 1, 0.42),
    (3, 2, -0.14),
    (3, 3, 0.06),
    (3, 4, 0.27),
    (3, 5, 0.16),
    (3, 6, 0.05),
    (3, 7, -0.10),
    (3, 8, 0.19),
    (3, 9, 0.0),
    (3, 10, 0.0),
    (3, 11, 0.0),
    (3, 12, 0.0),
    (4, 1, 0.44),
    (4, 2, -0.15),
    (4, 3, 0.07),
    (4, 4, 0.28),
    (4, 5, 0.17),
    (4, 6, 0.06),
    (4, 7, -0.11),
    (4, 8, 0.20),
    (4, 9, 0.0),
    (4, 10, 0.0),
    (4, 11, 0.0),
    (4, 12, 0.0);

INSERT INTO model_metrics (
    model_id,
    train_mse,
    validation_mse,
    test_mse,
    train_r2,
    validation_r2,
    test_r2
)
VALUES
    (1, 42.50, 78.20, 82.40, 0.91, 0.82, 0.80),
    (2, 45.10, 65.40, 67.80, 0.90, 0.86, 0.84),
    (3, 47.20, 68.10, 70.30, 0.89, 0.85, 0.83),
    (4, 45.80, 63.70, 66.10, 0.90, 0.87, 0.85);

INSERT INTO regularization_runs (
    experiment_id,
    family,
    alpha,
    l1_ratio,
    validation_mse,
    selected
)
VALUES
    (1, 'RIDGE', 0.01, NULL, 72.50, FALSE),
    (1, 'RIDGE', 0.10, NULL, 67.40, FALSE),
    (1, 'RIDGE', 0.80, NULL, 65.40, TRUE),
    (1, 'RIDGE', 3.00, NULL, 69.20, FALSE),
    (1, 'LASSO', 0.02, NULL, 70.10, FALSE),
    (1, 'LASSO', 0.08, NULL, 68.10, TRUE),
    (1, 'LASSO', 0.20, NULL, 74.50, FALSE),
    (1, 'ELASTIC_NET', 0.03, 0.5, 67.90, FALSE),
    (1, 'ELASTIC_NET', 0.08, 0.5, 63.70, TRUE),
    (1, 'ELASTIC_NET', 0.20, 0.5, 70.30, FALSE);

CREATE OR REPLACE VIEW model_sparsity AS
SELECT
    m.model_id,
    m.model_name,
    m.family,
    COUNT(c.feature_id) AS feature_count,
    COUNT(*) FILTER (WHERE c.is_zero) AS zero_coefficients,
    COUNT(*) FILTER (WHERE NOT c.is_zero) AS active_coefficients,
    ROUND(
        100.0 *
        COUNT(*) FILTER (WHERE c.is_zero) /
        NULLIF(COUNT(*), 0),
        2
    ) AS sparsity_percent
FROM models AS m
JOIN coefficients AS c
    ON c.model_id = m.model_id
GROUP BY
    m.model_id,
    m.model_name,
    m.family;

CREATE OR REPLACE VIEW model_comparison AS
SELECT
    m.model_name,
    m.family,
    m.alpha,
    m.l1_ratio,
    mm.train_mse,
    mm.validation_mse,
    mm.test_mse,
    mm.train_r2,
    mm.validation_r2,
    mm.test_r2
FROM models AS m
JOIN model_metrics AS mm
    ON mm.model_id = m.model_id;

CREATE OR REPLACE FUNCTION validate_model_configuration()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.family = 'ELASTIC_NET'
       AND (NEW.l1_ratio IS NULL
            OR NEW.l1_ratio < 0
            OR NEW.l1_ratio > 1) THEN
        RAISE EXCEPTION
            'Elastic Net requires l1_ratio in [0,1]';
    END IF;

    IF NEW.family <> 'ELASTIC_NET'
       AND NEW.l1_ratio IS NOT NULL THEN
        RAISE EXCEPTION
            'l1_ratio is only valid for Elastic Net';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_model_configuration
BEFORE INSERT OR UPDATE
ON models
FOR EACH ROW
EXECUTE FUNCTION validate_model_configuration();

CREATE OR REPLACE FUNCTION prevent_multiple_selected_runs()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.selected THEN
        UPDATE regularization_runs
        SET selected = FALSE
        WHERE experiment_id = NEW.experiment_id
          AND family = NEW.family
          AND run_id <> NEW.run_id;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_single_selected_run
BEFORE INSERT OR UPDATE
ON regularization_runs
FOR EACH ROW
EXECUTE FUNCTION prevent_multiple_selected_runs();

BEGIN;

INSERT INTO regularization_runs (
    experiment_id,
    family,
    alpha,
    l1_ratio,
    validation_mse,
    selected
)
VALUES (
    1,
    'RIDGE',
    1.2,
    NULL,
    64.8,
    TRUE
);

COMMIT;

SELECT
    model_name,
    family,
    alpha,
    l1_ratio,
    train_mse,
    validation_mse,
    test_mse,
    train_r2,
    validation_r2,
    test_r2
FROM model_comparison
ORDER BY test_mse;

SELECT
    model_name,
    family,
    feature_count,
    zero_coefficients,
    active_coefficients,
    sparsity_percent
FROM model_sparsity
ORDER BY sparsity_percent DESC;

SELECT
    m.model_name,
    f.feature_name,
    f.feature_role,
    c.coefficient_value,
    c.is_zero
FROM coefficients AS c
JOIN models AS m
    ON m.model_id = c.model_id
JOIN features AS f
    ON f.feature_id = c.feature_id
WHERE m.family IN ('RIDGE', 'LASSO', 'ELASTIC_NET')
ORDER BY
    m.model_name,
    ABS(c.coefficient_value) DESC;

SELECT
    family,
    alpha,
    l1_ratio,
    validation_mse,
    selected
FROM regularization_runs
WHERE experiment_id = 1
ORDER BY
    family,
    validation_mse;

SELECT
    feature_name,
    feature_role,
    COUNT(*) AS observation_count,
    AVG(
        CASE
            WHEN feature_name = 'temperature'
                THEN temperature
            WHEN feature_name = 'humidity'
                THEN humidity
            WHEN feature_name = 'pressure'
                THEN pressure
            WHEN feature_name = 'advertising_spend'
                THEN advertising_spend
            WHEN feature_name = 'discount_rate'
                THEN discount_rate
            WHEN feature_name = 'competitor_price'
                THEN competitor_price
            WHEN feature_name = 'store_traffic'
                THEN store_traffic
            ELSE NULL
        END
    ) AS average_value
FROM features
JOIN observations
    ON observations.experiment_id = features.experiment_id
WHERE features.experiment_id = 1
GROUP BY
    feature_name,
    feature_role
ORDER BY feature_name;

SELECT
    m.model_name,
    m.family,
    COUNT(*) FILTER (
        WHERE f.feature_role = 'NOISE'
          AND c.is_zero
    ) AS eliminated_noise_features,
    COUNT(*) FILTER (
        WHERE f.feature_role = 'CORRELATED_SIGNAL'
          AND NOT c.is_zero
    ) AS retained_correlated_features
FROM models AS m
JOIN coefficients AS c
    ON c.model_id = m.model_id
JOIN features AS f
    ON f.feature_id = c.feature_id
GROUP BY
    m.model_name,
    m.family
ORDER BY
    m.model_name;

DO $$
BEGIN
    BEGIN
        INSERT INTO models (
            experiment_id,
            model_name,
            family,
            alpha,
            l1_ratio,
            standardized_features,
            training_status
        )
        VALUES (
            1,
            'Invalid Elastic Net',
            'ELASTIC_NET',
            0.1,
            1.5,
            TRUE,
            'TRAINED'
        );
    EXCEPTION WHEN check_violation THEN
        RAISE NOTICE
            'Database correctly rejected an invalid Elastic Net ratio.';
    END;
END;
$$;

DO $$
BEGIN
    BEGIN
        INSERT INTO models (
            experiment_id,
            model_name,
            family,
            alpha,
            l1_ratio,
            standardized_features,
            training_status
        )
        VALUES (
            1,
            'Invalid Ridge',
            'RIDGE',
            0.1,
            0.5,
            TRUE,
            'TRAINED'
        );
    EXCEPTION WHEN check_violation THEN
        RAISE NOTICE
            'Database correctly rejected l1_ratio on Ridge.';
    END;
END;
$$;
