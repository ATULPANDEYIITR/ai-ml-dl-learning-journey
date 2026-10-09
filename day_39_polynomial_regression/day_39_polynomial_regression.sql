DROP SCHEMA IF EXISTS polynomial_regression_lab CASCADE;

CREATE SCHEMA polynomial_regression_lab;

SET search_path TO polynomial_regression_lab;

CREATE TABLE regression_datasets (
    dataset_id BIGSERIAL PRIMARY KEY,
    dataset_name TEXT NOT NULL UNIQUE,
    target_description TEXT NOT NULL,
    feature_description TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE observations (
    observation_id BIGSERIAL PRIMARY KEY,
    dataset_id BIGINT NOT NULL
        REFERENCES regression_datasets(dataset_id)
        ON DELETE CASCADE,
    feature_value NUMERIC(12, 5) NOT NULL,
    target_value NUMERIC(14, 5) NOT NULL,
    split TEXT NOT NULL,
    CONSTRAINT observations_split_ck
        CHECK (split IN ('TRAIN', 'TEST', 'VALIDATION')),
    CONSTRAINT observations_feature_finite_ck
        CHECK (feature_value = feature_value),
    CONSTRAINT observations_target_finite_ck
        CHECK (target_value = target_value)
);

CREATE INDEX observations_dataset_split_idx
    ON observations(dataset_id, split);

CREATE INDEX observations_dataset_feature_idx
    ON observations(dataset_id, feature_value);

CREATE TABLE polynomial_models (
    model_id BIGSERIAL PRIMARY KEY,
    dataset_id BIGINT NOT NULL
        REFERENCES regression_datasets(dataset_id)
        ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    polynomial_degree INTEGER NOT NULL,
    regularization_lambda NUMERIC(18, 8) NOT NULL DEFAULT 0,
    feature_scaling BOOLEAN NOT NULL DEFAULT TRUE,
    status TEXT NOT NULL DEFAULT 'CANDIDATE',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT polynomial_models_degree_ck
        CHECK (polynomial_degree >= 1 AND polynomial_degree <= 30),
    CONSTRAINT polynomial_models_lambda_ck
        CHECK (regularization_lambda >= 0),
    CONSTRAINT polynomial_models_status_ck
        CHECK (
            status IN (
                'CANDIDATE',
                'VALIDATED',
                'REJECTED',
                'APPROVED',
                'RETIRED'
            )
        ),
    UNIQUE (dataset_id, model_name)
);

CREATE TABLE model_coefficients (
    model_id BIGINT NOT NULL
        REFERENCES polynomial_models(model_id)
        ON DELETE CASCADE,
    polynomial_power INTEGER NOT NULL,
    coefficient NUMERIC(30, 12) NOT NULL,
    PRIMARY KEY (model_id, polynomial_power),
    CONSTRAINT model_coefficients_power_ck
        CHECK (polynomial_power >= 0)
);

CREATE TABLE model_evaluations (
    evaluation_id BIGSERIAL PRIMARY KEY,
    model_id BIGINT NOT NULL
        REFERENCES polynomial_models(model_id)
        ON DELETE CASCADE,
    evaluation_type TEXT NOT NULL,
    mae NUMERIC(18, 8) NOT NULL,
    rmse NUMERIC(18, 8) NOT NULL,
    r_squared NUMERIC(18, 8) NOT NULL,
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT model_evaluations_type_ck
        CHECK (
            evaluation_type IN (
                'TRAIN',
                'VALIDATION',
                'TEST',
                'CROSS_VALIDATION'
            )
        ),
    CONSTRAINT model_evaluations_mae_ck
        CHECK (mae >= 0),
    CONSTRAINT model_evaluations_rmse_ck
        CHECK (rmse >= 0),
    CONSTRAINT model_evaluations_r2_ck
        CHECK (r_squared <= 1)
);

CREATE INDEX model_evaluations_model_type_idx
    ON model_evaluations(model_id, evaluation_type);

CREATE TABLE cross_validation_results (
    cv_result_id BIGSERIAL PRIMARY KEY,
    model_id BIGINT NOT NULL
        REFERENCES polynomial_models(model_id)
        ON DELETE CASCADE,
    fold_number INTEGER NOT NULL,
    validation_rmse NUMERIC(18, 8) NOT NULL,
    validation_mae NUMERIC(18, 8) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT cv_fold_ck
        CHECK (fold_number >= 1),
    CONSTRAINT cv_rmse_ck
        CHECK (validation_rmse >= 0),
    CONSTRAINT cv_mae_ck
        CHECK (validation_mae >= 0),
    UNIQUE (model_id, fold_number)
);

CREATE TABLE model_governance_policies (
    policy_id BIGSERIAL PRIMARY KEY,
    policy_name TEXT NOT NULL UNIQUE,
    maximum_degree INTEGER NOT NULL,
    maximum_test_rmse NUMERIC(18, 8) NOT NULL,
    minimum_test_r_squared NUMERIC(18, 8) NOT NULL,
    maximum_train_test_rmse_gap NUMERIC(18, 8) NOT NULL,
    minimum_approval_reviews INTEGER NOT NULL DEFAULT 1,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    CONSTRAINT governance_degree_ck
        CHECK (maximum_degree >= 1),
    CONSTRAINT governance_test_rmse_ck
        CHECK (maximum_test_rmse > 0),
    CONSTRAINT governance_r2_ck
        CHECK (minimum_test_r_squared BETWEEN -1 AND 1),
    CONSTRAINT governance_gap_ck
        CHECK (maximum_train_test_rmse_gap >= 0),
    CONSTRAINT governance_reviews_ck
        CHECK (minimum_approval_reviews >= 0)
);

CREATE TABLE model_policy_decisions (
    decision_id BIGSERIAL PRIMARY KEY,
    model_id BIGINT NOT NULL
        REFERENCES polynomial_models(model_id)
        ON DELETE CASCADE,
    policy_id BIGINT NOT NULL
        REFERENCES model_governance_policies(policy_id)
        ON DELETE RESTRICT,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    decided_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT policy_decision_type_ck
        CHECK (decision IN ('APPROVE', 'REJECT', 'HOLD'))
);

CREATE TABLE reviewers (
    reviewer_id BIGSERIAL PRIMARY KEY,
    reviewer_name TEXT NOT NULL UNIQUE,
    role_name TEXT NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE model_reviews (
    review_id BIGSERIAL PRIMARY KEY,
    model_id BIGINT NOT NULL
        REFERENCES polynomial_models(model_id)
        ON DELETE CASCADE,
    reviewer_id BIGINT NOT NULL
        REFERENCES reviewers(reviewer_id)
        ON DELETE RESTRICT,
    review_state TEXT NOT NULL,
    review_comment TEXT NOT NULL,
    reviewed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT model_review_state_ck
        CHECK (
            review_state IN (
                'APPROVED',
                'CHANGES_REQUESTED',
                'COMMENTED'
            )
        )
);

CREATE INDEX model_reviews_model_idx
    ON model_reviews(model_id);

CREATE TABLE prediction_requests (
    request_id BIGSERIAL PRIMARY KEY,
    model_id BIGINT NOT NULL
        REFERENCES polynomial_models(model_id)
        ON DELETE RESTRICT,
    feature_value NUMERIC(12, 5) NOT NULL,
    request_status TEXT NOT NULL DEFAULT 'PENDING',
    requested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT prediction_status_ck
        CHECK (
            request_status IN (
                'PENDING',
                'COMPLETED',
                'REJECTED',
                'EXTRAPOLATION_WARNING'
            )
        )
);

INSERT INTO regression_datasets (
    dataset_name,
    target_description,
    feature_description
)
VALUES (
    'energy_demand_temperature',
    'Synthetic electricity demand affected by heating and cooling load',
    'Outdoor temperature in degrees Celsius'
);

INSERT INTO model_governance_policies (
    policy_name,
    maximum_degree,
    maximum_test_rmse,
    minimum_test_r_squared,
    maximum_train_test_rmse_gap,
    minimum_approval_reviews
)
VALUES (
    'production_polynomial_policy',
    6,
    35.0,
    0.85,
    15.0,
    2
);

INSERT INTO reviewers (
    reviewer_name,
    role_name
)
VALUES
    ('Priya Sharma', 'Data Science Lead'),
    ('Rohan Mehta', 'Energy Analytics Lead'),
    ('Neha Singh', 'Model Risk Reviewer');

WITH dataset AS (
    SELECT dataset_id
    FROM regression_datasets
    WHERE dataset_name = 'energy_demand_temperature'
),
generated AS (
    SELECT
        generate_series(1, 120) AS observation_number
)
INSERT INTO observations (
    dataset_id,
    feature_value,
    target_value,
    split
)
SELECT
    dataset.dataset_id,
    -5 + (45.0 * (observation_number - 1) / 119.0),
    420
        - 13 * (-5 + (45.0 * (observation_number - 1) / 119.0))
        + 0.72 * power(
            -5 + (45.0 * (observation_number - 1) / 119.0),
            2
        )
        - 0.012 * power(
            -5 + (45.0 * (observation_number - 1) / 119.0),
            3
        ),
    CASE
        WHEN observation_number % 5 = 0 THEN 'TEST'
        ELSE 'TRAIN'
    END
FROM dataset
CROSS JOIN generated;

INSERT INTO polynomial_models (
    dataset_id,
    model_name,
    polynomial_degree,
    regularization_lambda,
    feature_scaling,
    status
)
SELECT
    dataset_id,
    'linear_baseline',
    1,
    0,
    TRUE,
    'CANDIDATE'
FROM regression_datasets
WHERE dataset_name = 'energy_demand_temperature';

INSERT INTO polynomial_models (
    dataset_id,
    model_name,
    polynomial_degree,
    regularization_lambda,
    feature_scaling,
    status
)
SELECT
    dataset_id,
    'quadratic_production',
    2,
    0.1,
    TRUE,
    'VALIDATED'
FROM regression_datasets
WHERE dataset_name = 'energy_demand_temperature';

INSERT INTO polynomial_models (
    dataset_id,
    model_name,
    polynomial_degree,
    regularization_lambda,
    feature_scaling,
    status
)
SELECT
    dataset_id,
    'high_degree_risk_candidate',
    10,
    0.001,
    TRUE,
    'CANDIDATE'
FROM regression_datasets
WHERE dataset_name = 'energy_demand_temperature';

WITH models AS (
    SELECT model_id, model_name
    FROM polynomial_models
)
INSERT INTO model_coefficients (
    model_id,
    polynomial_power,
    coefficient
)
SELECT
    model_id,
    power,
    CASE
        WHEN model_name = 'quadratic_production' AND power = 0 THEN 420
        WHEN model_name = 'quadratic_production' AND power = 1 THEN -13
        WHEN model_name = 'quadratic_production' AND power = 2 THEN 0.72
        ELSE 0
    END
FROM models
CROSS JOIN generate_series(
    0,
    CASE
        WHEN model_name = 'quadratic_production' THEN 2
        ELSE 10
    END
) AS power;

INSERT INTO model_evaluations (
    model_id,
    evaluation_type,
    mae,
    rmse,
    r_squared
)
SELECT
    model_id,
    'TRAIN',
    CASE model_name
        WHEN 'linear_baseline' THEN 81.4
        WHEN 'quadratic_production' THEN 17.8
        ELSE 3.2
    END,
    CASE model_name
        WHEN 'linear_baseline' THEN 96.1
        WHEN 'quadratic_production' THEN 22.7
        ELSE 8.4
    END,
    CASE model_name
        WHEN 'linear_baseline' THEN 0.63
        WHEN 'quadratic_production' THEN 0.96
        ELSE 0.995
    END
FROM polynomial_models;

INSERT INTO model_evaluations (
    model_id,
    evaluation_type,
    mae,
    rmse,
    r_squared
)
SELECT
    model_id,
    'TEST',
    CASE model_name
        WHEN 'linear_baseline' THEN 83.2
        WHEN 'quadratic_production' THEN 23.4
        ELSE 61.7
    END,
    CASE model_name
        WHEN 'linear_baseline' THEN 98.2
        WHEN 'quadratic_production' THEN 28.1
        ELSE 79.5
    END,
    CASE model_name
        WHEN 'linear_baseline' THEN 0.61
        WHEN 'quadratic_production' THEN 0.94
        ELSE 0.48
    END
FROM polynomial_models;

INSERT INTO cross_validation_results (
    model_id,
    fold_number,
    validation_rmse,
    validation_mae
)
SELECT
    model_id,
    fold_number,
    CASE model_name
        WHEN 'linear_baseline'
            THEN 97 + fold_number * 0.7
        WHEN 'quadratic_production'
            THEN 28 + fold_number * 0.3
        ELSE 74 + fold_number * 1.2
    END,
    CASE model_name
        WHEN 'linear_baseline'
            THEN 82 + fold_number * 0.4
        WHEN 'quadratic_production'
            THEN 22 + fold_number * 0.2
        ELSE 59 + fold_number * 0.8
    END
FROM polynomial_models
CROSS JOIN generate_series(1, 5) AS fold_number;

INSERT INTO model_reviews (
    model_id,
    reviewer_id,
    review_state,
    review_comment
)
SELECT
    model.model_id,
    reviewer.reviewer_id,
    'APPROVED',
    CASE
        WHEN reviewer.reviewer_name = 'Priya Sharma'
            THEN 'Validation metrics meet the production policy.'
        ELSE
            'Polynomial degree and regularization are appropriate for the observed range.'
    END
FROM polynomial_models model
CROSS JOIN reviewers reviewer
WHERE model.model_name = 'quadratic_production'
AND reviewer.reviewer_name IN (
    'Priya Sharma',
    'Rohan Mehta'
);

INSERT INTO model_reviews (
    model_id,
    reviewer_id,
    review_state,
    review_comment
)
SELECT
    model.model_id,
    reviewer.reviewer_id,
    'CHANGES_REQUESTED',
    'High train-test error divergence indicates probable overfitting.'
FROM polynomial_models model
JOIN reviewers reviewer
    ON reviewer.reviewer_name = 'Neha Singh'
WHERE model.model_name = 'high_degree_risk_candidate';

WITH evaluation_summary AS (
    SELECT
        model_id,
        MAX(
            rmse
        ) FILTER (
            WHERE evaluation_type = 'TRAIN'
        ) AS train_rmse,
        MAX(
            rmse
        ) FILTER (
            WHERE evaluation_type = 'TEST'
        ) AS test_rmse,
        MAX(
            r_squared
        ) FILTER (
            WHERE evaluation_type = 'TEST'
        ) AS test_r_squared
    FROM model_evaluations
    GROUP BY model_id
)
SELECT
    model.model_name,
    model.polynomial_degree,
    summary.train_rmse,
    summary.test_rmse,
    summary.test_r_squared,
    summary.test_rmse - summary.train_rmse
        AS train_test_rmse_gap
FROM polynomial_models model
JOIN evaluation_summary summary
    ON summary.model_id = model.model_id
ORDER BY summary.test_rmse;

WITH cv_summary AS (
    SELECT
        model_id,
        AVG(validation_rmse) AS mean_cv_rmse,
        STDDEV_POP(validation_rmse) AS cv_rmse_stddev
    FROM cross_validation_results
    GROUP BY model_id
)
SELECT
    model.model_name,
    model.polynomial_degree,
    cv.mean_cv_rmse,
    cv.cv_rmse_stddev
FROM polynomial_models model
JOIN cv_summary cv
    ON cv.model_id = model.model_id
ORDER BY cv.mean_cv_rmse;

WITH test_metrics AS (
    SELECT
        model_id,
        MAX(rmse) AS test_rmse,
        MAX(r_squared) AS test_r_squared
    FROM model_evaluations
    WHERE evaluation_type = 'TEST'
    GROUP BY model_id
),
review_counts AS (
    SELECT
        model_id,
        COUNT(*) FILTER (
            WHERE review_state = 'APPROVED'
        ) AS approvals,
        COUNT(*) FILTER (
            WHERE review_state = 'CHANGES_REQUESTED'
        ) AS requested_changes
    FROM model_reviews
    GROUP BY model_id
)
SELECT
    model.model_name,
    model.polynomial_degree,
    test_metrics.test_rmse,
    test_metrics.test_r_squared,
    COALESCE(review_counts.approvals, 0) AS approvals,
    COALESCE(review_counts.requested_changes, 0)
        AS requested_changes,
    CASE
        WHEN model.polynomial_degree <= policy.maximum_degree
        AND test_metrics.test_rmse <= policy.maximum_test_rmse
        AND test_metrics.test_r_squared >= policy.minimum_test_r_squared
        AND COALESCE(review_counts.approvals, 0)
            >= policy.minimum_approval_reviews
        AND COALESCE(review_counts.requested_changes, 0) = 0
        THEN 'ELIGIBLE'
        ELSE 'NOT_ELIGIBLE'
    END AS merge_style_model_approval
FROM polynomial_models model
CROSS JOIN model_governance_policies policy
JOIN test_metrics
    ON test_metrics.model_id = model.model_id
LEFT JOIN review_counts
    ON review_counts.model_id = model.model_id
WHERE policy.active = TRUE;

CREATE VIEW approved_model_catalog AS
SELECT
    model.model_id,
    model.model_name,
    model.polynomial_degree,
    model.regularization_lambda,
    model.feature_scaling,
    model.status,
    MAX(
        evaluations.rmse
    ) FILTER (
        WHERE evaluations.evaluation_type = 'TEST'
    ) AS test_rmse,
    MAX(
        evaluations.r_squared
    ) FILTER (
        WHERE evaluations.evaluation_type = 'TEST'
    ) AS test_r_squared
FROM polynomial_models model
LEFT JOIN model_evaluations evaluations
    ON evaluations.model_id = model.model_id
WHERE model.status = 'APPROVED'
GROUP BY
    model.model_id,
    model.model_name,
    model.polynomial_degree,
    model.regularization_lambda,
    model.feature_scaling,
    model.status;

CREATE OR REPLACE FUNCTION enforce_model_approval_rules()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    test_rmse NUMERIC;
    test_r2 NUMERIC;
    training_rmse NUMERIC;
    approved_reviews INTEGER;
    requested_changes INTEGER;
    maximum_degree INTEGER;
    maximum_test_rmse NUMERIC;
    minimum_test_r2 NUMERIC;
    maximum_gap NUMERIC;
    required_reviews INTEGER;
BEGIN
    IF NEW.status <> 'APPROVED' THEN
        RETURN NEW;
    END IF;

    SELECT
        maximum_degree,
        maximum_test_rmse,
        minimum_test_r_squared,
        maximum_train_test_rmse_gap,
        minimum_approval_reviews
    INTO
        maximum_degree,
        maximum_test_rmse,
        minimum_test_r2,
        maximum_gap,
        required_reviews
    FROM model_governance_policies
    WHERE active = TRUE
    ORDER BY policy_id
    LIMIT 1;

    SELECT rmse
    INTO test_rmse
    FROM model_evaluations
    WHERE model_id = NEW.model_id
      AND evaluation_type = 'TEST'
    ORDER BY evaluated_at DESC
    LIMIT 1;

    SELECT r_squared
    INTO test_r2
    FROM model_evaluations
    WHERE model_id = NEW.model_id
      AND evaluation_type = 'TEST'
    ORDER BY evaluated_at DESC
    LIMIT 1;

    SELECT rmse
    INTO training_rmse
    FROM model_evaluations
    WHERE model_id = NEW.model_id
      AND evaluation_type = 'TRAIN'
    ORDER BY evaluated_at DESC
    LIMIT 1;

    SELECT COUNT(*)
    INTO approved_reviews
    FROM model_reviews
    WHERE model_id = NEW.model_id
      AND review_state = 'APPROVED';

    SELECT COUNT(*)
    INTO requested_changes
    FROM model_reviews
    WHERE model_id = NEW.model_id
      AND review_state = 'CHANGES_REQUESTED';

    IF NEW.polynomial_degree > maximum_degree THEN
        RAISE EXCEPTION
            'Model degree % exceeds governance maximum %',
            NEW.polynomial_degree,
            maximum_degree;
    END IF;

    IF test_rmse IS NULL
       OR test_rmse > maximum_test_rmse THEN
        RAISE EXCEPTION
            'Test RMSE % fails governance threshold %',
            test_rmse,
            maximum_test_rmse;
    END IF;

    IF test_r2 IS NULL
       OR test_r2 < minimum_test_r2 THEN
        RAISE EXCEPTION
            'Test R-squared % fails governance threshold %',
            test_r2,
            minimum_test_r2;
    END IF;

    IF training_rmse IS NULL
       OR test_rmse - training_rmse > maximum_gap THEN
        RAISE EXCEPTION
            'Train-test RMSE gap fails governance policy';
    END IF;

    IF approved_reviews < required_reviews THEN
        RAISE EXCEPTION
            'Model requires % approvals but has %',
            required_reviews,
            approved_reviews;
    END IF;

    IF requested_changes > 0 THEN
        RAISE EXCEPTION
            'Model has unresolved requested changes';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER model_approval_governance_trigger
BEFORE UPDATE OF status
ON polynomial_models
FOR EACH ROW
EXECUTE FUNCTION enforce_model_approval_rules();

BEGIN;

UPDATE polynomial_models
SET status = 'APPROVED'
WHERE model_name = 'quadratic_production';

COMMIT;

BEGIN;

UPDATE polynomial_models
SET status = 'APPROVED'
WHERE model_name = 'high_degree_risk_candidate';

ROLLBACK;

INSERT INTO prediction_requests (
    model_id,
    feature_value,
    request_status
)
SELECT
    model_id,
    feature_value,
    CASE
        WHEN feature_value BETWEEN
            (
                SELECT MIN(observation.feature_value)
                FROM observations observation
                WHERE observation.dataset_id = model.dataset_id
            )
            AND
            (
                SELECT MAX(observation.feature_value)
                FROM observations observation
                WHERE observation.dataset_id = model.dataset_id
            )
        THEN 'PENDING'
        ELSE 'EXTRAPOLATION_WARNING'
    END
FROM polynomial_models model
CROSS JOIN (
    VALUES
        (-5.0::NUMERIC),
        (10.0::NUMERIC),
        (25.0::NUMERIC),
        (40.0::NUMERIC),
        (65.0::NUMERIC)
) AS requested(feature_value)
WHERE model.model_name = 'quadratic_production';

SELECT
    request_id,
    feature_value,
    request_status
FROM prediction_requests
ORDER BY request_id;

SELECT
    model_name,
    polynomial_degree,
    regularization_lambda,
    status,
    test_rmse,
    test_r_squared
FROM approved_model_catalog
ORDER BY test_rmse;

SELECT
    model.model_name,
    coefficient.polynomial_power,
    coefficient.coefficient
FROM polynomial_models model
JOIN model_coefficients coefficient
    ON coefficient.model_id = model.model_id
WHERE model.model_name = 'quadratic_production'
ORDER BY coefficient.polynomial_power;
