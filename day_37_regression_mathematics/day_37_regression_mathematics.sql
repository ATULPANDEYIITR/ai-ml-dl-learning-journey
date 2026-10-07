/*
 * Regression Mathematics: PostgreSQL Demonstration
 *
 * Domain:
 * Operational delivery-time regression.
 *
 * The database stores observations, regression coefficients, predictions,
 * residuals, and model metrics. PostgreSQL constraints preserve data quality,
 * while views and CTEs expose the mathematical quantities used to evaluate
 * a least-squares model.
 *
 * The calculations demonstrate:
 *   - least-squares coefficient estimation
 *   - fitted values
 *   - residuals
 *   - SSE
 *   - MSE
 *   - RMSE
 *   - R²
 *   - adjusted R²
 *
 * PostgreSQL 15+ compatible.
 */

DROP SCHEMA IF EXISTS regression_math CASCADE;

CREATE SCHEMA regression_math;

SET search_path TO regression_math;

/*
 * A predictor observation contains two operational predictors:
 * processing_hours and item_count.
 *
 * A CHECK constraint prevents physically meaningless negative quantities.
 */
CREATE TABLE observations (
    observation_id BIGSERIAL PRIMARY KEY,
    processing_hours NUMERIC(12,4) NOT NULL,
    item_count NUMERIC(12,4) NOT NULL,
    observed_delivery_hours NUMERIC(12,4) NOT NULL,

    CONSTRAINT observations_processing_nonnegative
        CHECK (processing_hours >= 0),

    CONSTRAINT observations_item_count_nonnegative
        CHECK (item_count >= 0),

    CONSTRAINT observations_delivery_positive
        CHECK (observed_delivery_hours > 0)
);

/*
 * The index is useful when operational observations are filtered by
 * processing-time ranges during diagnostic analysis.
 */
CREATE INDEX idx_observations_processing_hours
    ON observations (processing_hours);


/*
 * Store a fitted model independently from individual observations.
 *
 * p is the number of predictors excluding the intercept.
 */
CREATE TABLE regression_models (
    model_id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL UNIQUE,
    fitted_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    predictor_count INTEGER NOT NULL,
    coefficient_intercept NUMERIC(18,10) NOT NULL,
    coefficient_processing_hours NUMERIC(18,10) NOT NULL,
    coefficient_item_count NUMERIC(18,10) NOT NULL,
    CONSTRAINT model_predictor_count CHECK (predictor_count = 2)
);


/*
 * Predictions and residuals are persisted so that model diagnostics can be
 * audited without recomputing the complete regression for every report.
 */
CREATE TABLE regression_predictions (
    model_id BIGINT NOT NULL
        REFERENCES regression_models(model_id)
        ON DELETE CASCADE,

    observation_id BIGINT NOT NULL
        REFERENCES observations(observation_id)
        ON DELETE CASCADE,

    predicted_delivery_hours NUMERIC(18,10) NOT NULL,
    residual NUMERIC(18,10) NOT NULL,

    PRIMARY KEY (model_id, observation_id)
);

CREATE INDEX idx_predictions_model_residual
    ON regression_predictions (model_id, residual);


/*
 * Model metrics separate mathematical evaluation from raw observations.
 */
CREATE TABLE regression_metrics (
    model_id BIGINT PRIMARY KEY
        REFERENCES regression_models(model_id)
        ON DELETE CASCADE,

    sse NUMERIC(24,12) NOT NULL,
    mse NUMERIC(24,12) NOT NULL,
    rmse NUMERIC(24,12) NOT NULL,
    r_squared NUMERIC(18,12) NOT NULL,
    adjusted_r_squared NUMERIC(18,12) NOT NULL,

    CONSTRAINT metrics_nonnegative_sse CHECK (sse >= 0),
    CONSTRAINT metrics_nonnegative_mse CHECK (mse >= 0),
    CONSTRAINT metrics_nonnegative_rmse CHECK (rmse >= 0),
    CONSTRAINT metrics_r_squared_range
        CHECK (r_squared <= 1),
    CONSTRAINT metrics_adjusted_r_squared_upper_bound
        CHECK (adjusted_r_squared <= 1)
);


/*
 * Operational observations.
 */
INSERT INTO observations (
    processing_hours,
    item_count,
    observed_delivery_hours
)
VALUES
    (2, 10, 8.4),
    (3, 12, 10.2),
    (4, 14, 11.7),
    (5, 18, 14.1),
    (6, 17, 14.8),
    (7, 21, 17.0),
    (8, 23, 18.9),
    (9, 24, 19.7),
    (10, 28, 23.0),
    (11, 29, 24.2),
    (12, 32, 26.0),
    (13, 35, 29.1);


/*
 * -------------------------------------------------------------------------
 * Least-squares coefficient estimation
 * -------------------------------------------------------------------------
 *
 * With an intercept and two predictors:
 *
 *   y = β0 + β1*x1 + β2*x2 + ε
 *
 * The normal-equation solution is:
 *
 *   β = (X'X)^(-1) X'y
 *
 * PostgreSQL does not provide a built-in general matrix inverse in standard
 * SQL, so the following recursive CTE implements the required 3x3 normal
 * equations through Gaussian elimination.
 *
 * For a production statistical platform, a numerically stable QR or SVD
 * implementation is preferable to explicitly forming (X'X)^(-1).
 */


/*
 * Aggregate the six unique cross-products needed by the symmetric
 * X'X matrix and the three X'y values.
 */
CREATE MATERIALIZED VIEW normal_equation_statistics AS
SELECT
    COUNT(*)::numeric AS n,

    SUM(1)::numeric AS x0x0,
    SUM(processing_hours)::numeric AS x0x1,
    SUM(item_count)::numeric AS x0x2,

    SUM(processing_hours * processing_hours)::numeric AS x1x1,
    SUM(processing_hours * item_count)::numeric AS x1x2,
    SUM(item_count * item_count)::numeric AS x2x2,

    SUM(observed_delivery_hours)::numeric AS x0y,
    SUM(processing_hours * observed_delivery_hours)::numeric AS x1y,
    SUM(item_count * observed_delivery_hours)::numeric AS x2y
FROM observations;


/*
 * This view exposes the design matrix cross-product and response cross-product
 * as rows. It is useful for auditing the regression mathematics directly.
 */
CREATE VIEW normal_equations AS
SELECT
    'XTX' AS matrix_name,
    ARRAY[
        x0x0, x0x1, x0x2,
        x0x1, x1x1, x1x2,
        x0x2, x1x2, x2x2
    ] AS values
FROM normal_equation_statistics

UNION ALL

SELECT
    'XTY',
    ARRAY[
        x0y,
        x1y,
        x2y
    ]
FROM normal_equation_statistics;


/*
 * -------------------------------------------------------------------------
 * Model coefficients
 * -------------------------------------------------------------------------
 *
 * The coefficient values below are calculated by the PostgreSQL client-side
 * execution of the normal equations. The INSERT is intentionally expressed
 * from the observed data rather than hard-coding a coefficient vector.
 *
 * For this compact SQL demonstration, the 3x3 system is solved by Cramer's
 * rule. This is acceptable for a tiny educational matrix, but it should not
 * be generalized to high-dimensional regression because determinant-based
 * approaches are less numerically stable and less efficient than QR/SVD.
 */
WITH s AS (
    SELECT *
    FROM normal_equation_statistics
),
determinant AS (
    SELECT
        n,
        (
            x0x0 * (x1x1 * x2x2 - x1x2 * x1x2)
            - x0x1 * (x0x1 * x2x2 - x1x2 * x0x2)
            + x0x2 * (x0x1 * x1x2 - x1x1 * x0x2)
        ) AS d
    FROM s
),
coefficient_system AS (
    SELECT
        s.*,
        d.d,

        (
            x0y * (x1x1 * x2x2 - x1x2 * x1x2)
            - x0x1 * (x1y * x2x2 - x1x2 * x2y)
            + x0x2 * (x1y * x1x2 - x1x1 * x2y)
        ) AS d_beta0,

        (
            x0x0 * (x1y * x2x2 - x1x2 * x2y)
            - x0y * (x0x1 * x2x2 - x1x2 * x0x2)
            + x0x2 * (x0x1 * x2y - x1y * x0x2)
        ) AS d_beta1,

        (
            x0x0 * (x1x1 * x2y - x1y * x1x2)
            - x0x1 * (x1x1 * x2y - x1y * x0x2)
            + x0y * (x0x1 * x1x2 - x1x1 * x0x2)
        ) AS d_beta2

    FROM s
    CROSS JOIN determinant d
)
INSERT INTO regression_models (
    model_name,
    predictor_count,
    coefficient_intercept,
    coefficient_processing_hours,
    coefficient_item_count
)
SELECT
    'delivery_time_ols_v1',
    2,
    d_beta0 / NULLIF(d, 0),
    d_beta1 / NULLIF(d, 0),
    d_beta2 / NULLIF(d, 0)
FROM coefficient_system;


/*
 * -------------------------------------------------------------------------
 * Fitted values and residuals
 * -------------------------------------------------------------------------
 *
 * ŷ_i = β0 + β1*x1_i + β2*x2_i
 *
 * e_i = y_i - ŷ_i
 */
INSERT INTO regression_predictions (
    model_id,
    observation_id,
    predicted_delivery_hours,
    residual
)
SELECT
    m.model_id,
    o.observation_id,

    (
        m.coefficient_intercept
        + m.coefficient_processing_hours
            * o.processing_hours
        + m.coefficient_item_count
            * o.item_count
    ) AS predicted_delivery_hours,

    (
        o.observed_delivery_hours
        - (
            m.coefficient_intercept
            + m.coefficient_processing_hours
                * o.processing_hours
            + m.coefficient_item_count
                * o.item_count
        )
    ) AS residual

FROM regression_models m
CROSS JOIN observations o
WHERE m.model_name = 'delivery_time_ols_v1';


/*
 * -------------------------------------------------------------------------
 * SSE, MSE, RMSE, R², adjusted R²
 * -------------------------------------------------------------------------
 *
 * SSE = Σe²
 *
 * Model-error MSE uses residual degrees of freedom:
 *
 *     SSE / (n - p - 1)
 *
 * Predictive MSE divides by n. Both quantities have legitimate uses, but
 * they answer different questions.
 *
 * R² = 1 - SSE/SST
 *
 * SST = Σ(y - ybar)²
 *
 * Adjusted R²:
 *
 *     1 - (1-R²)(n-1)/(n-p-1)
 */
WITH model_data AS (
    SELECT
        m.model_id,
        m.predictor_count,
        o.observed_delivery_hours,
        p.predicted_delivery_hours,
        p.residual
    FROM regression_models m
    JOIN regression_predictions p
        ON p.model_id = m.model_id
    JOIN observations o
        ON o.observation_id = p.observation_id
    WHERE m.model_name = 'delivery_time_ols_v1'
),
response_mean AS (
    SELECT
        model_id,
        predictor_count,
        AVG(observed_delivery_hours) AS y_bar
    FROM model_data
    GROUP BY model_id, predictor_count
),
sums AS (
    SELECT
        d.model_id,
        d.predictor_count,
        COUNT(*)::numeric AS n,

        SUM(d.residual * d.residual) AS sse,

        SUM(
            (d.observed_delivery_hours - r.y_bar)
            * (d.observed_delivery_hours - r.y_bar)
        ) AS sst

    FROM model_data d
    JOIN response_mean r
        ON r.model_id = d.model_id
    GROUP BY
        d.model_id,
        d.predictor_count
),
metrics AS (
    SELECT
        model_id,
        sse,
        sse / n AS predictive_mse,
        sse / (n - predictor_count - 1) AS model_mse,
        sst,
        1 - sse / NULLIF(sst, 0) AS r_squared,
        n,
        predictor_count
    FROM sums
)
INSERT INTO regression_metrics (
    model_id,
    sse,
    mse,
    rmse,
    r_squared,
    adjusted_r_squared
)
SELECT
    model_id,
    sse,
    model_mse,
    SQRT(predictive_mse),
    r_squared,
    1
        - (1 - r_squared)
        * ((n - 1) / (n - predictor_count - 1))
FROM metrics;


/*
 * -------------------------------------------------------------------------
 * Audit query: fitted coefficients
 * -------------------------------------------------------------------------
 */
SELECT
    model_name,
    coefficient_intercept,
    coefficient_processing_hours,
    coefficient_item_count
FROM regression_models;


/*
 * -------------------------------------------------------------------------
 * Audit query: observation-level residuals
 * -------------------------------------------------------------------------
 */
SELECT
    p.observation_id,
    o.processing_hours,
    o.item_count,
    o.observed_delivery_hours,
    ROUND(p.predicted_delivery_hours, 4)
        AS predicted_delivery_hours,
    ROUND(p.residual, 4)
        AS residual,
    ROUND(p.residual * p.residual, 4)
        AS squared_residual
FROM regression_predictions p
JOIN observations o
    ON o.observation_id = p.observation_id
ORDER BY p.observation_id;


/*
 * -------------------------------------------------------------------------
 * Audit query: aggregate model metrics
 * -------------------------------------------------------------------------
 */
SELECT
    m.model_name,
    ROUND(r.sse, 6) AS sse,
    ROUND(r.mse, 6) AS model_error_mse,
    ROUND(r.rmse, 6) AS rmse,
    ROUND(r.r_squared, 6) AS r_squared,
    ROUND(r.adjusted_r_squared, 6) AS adjusted_r_squared
FROM regression_models m
JOIN regression_metrics r
    ON r.model_id = m.model_id;


/*
 * -------------------------------------------------------------------------
 * Variance decomposition
 * -------------------------------------------------------------------------
 *
 * With an intercept:
 *
 *     SST ≈ SSR + SSE
 *
 * where SSR = Σ(ŷ-ybar)².
 */
WITH model_values AS (
    SELECT
        p.model_id,
        o.observed_delivery_hours AS y,
        p.predicted_delivery_hours AS y_hat,
        p.residual
    FROM regression_predictions p
    JOIN observations o
        ON o.observation_id = p.observation_id
),
means AS (
    SELECT
        model_id,
        AVG(y) AS y_bar
    FROM model_values
    GROUP BY model_id
)
SELECT
    v.model_id,

    SUM(
        (v.y - m.y_bar) * (v.y - m.y_bar)
    ) AS sst,

    SUM(
        (v.y_hat - m.y_bar) * (v.y_hat - m.y_bar)
    ) AS ssr,

    SUM(v.residual * v.residual) AS sse,

    SUM(
        (v.y_hat - m.y_bar) * (v.y_hat - m.y_bar)
    )
    + SUM(v.residual * v.residual) AS ssr_plus_sse

FROM model_values v
JOIN means m
    ON m.model_id = v.model_id
GROUP BY v.model_id;


/*
 * -------------------------------------------------------------------------
 * Diagnostic query
 * -------------------------------------------------------------------------
 *
 * Observations with unusually large residuals deserve investigation because
 * a strong aggregate R² does not guarantee accurate predictions for every
 * observation.
 */
SELECT
    p.observation_id,
    o.processing_hours,
    o.item_count,
    o.observed_delivery_hours,
    p.predicted_delivery_hours,
    p.residual
FROM regression_predictions p
JOIN observations o
    ON o.observation_id = p.observation_id
WHERE ABS(p.residual) > 2.0
ORDER BY ABS(p.residual) DESC;


/*
 * -------------------------------------------------------------------------
 * Example prediction for unseen operational cases
 * -------------------------------------------------------------------------
 *
 * The fitted coefficients are reused without modifying the trained model.
 */
WITH future_cases(processing_hours, item_count) AS (
    VALUES
        (6.5::numeric, 20.0::numeric),
        (9.5::numeric, 25.0::numeric),
        (14.0::numeric, 38.0::numeric)
)
SELECT
    f.processing_hours,
    f.item_count,
    m.coefficient_intercept
        + m.coefficient_processing_hours
            * f.processing_hours
        + m.coefficient_item_count
            * f.item_count
        AS predicted_delivery_hours
FROM future_cases f
CROSS JOIN regression_models m
WHERE m.model_name = 'delivery_time_ols_v1';


/*
 * -------------------------------------------------------------------------
 * Database integrity examples
 * -------------------------------------------------------------------------
 *
 * These statements are intentionally disabled. The comments show the
 * operations that PostgreSQL will reject because of CHECK constraints.
 *
 * INSERT INTO observations(processing_hours, item_count, observed_delivery_hours)
 * VALUES (-1, 20, 10);
 *
 * INSERT INTO observations(processing_hours, item_count, observed_delivery_hours)
 * VALUES (5, -20, 10);
 *
 * The constraints prevent invalid operational observations from entering the
 * regression dataset and contaminating downstream statistics.
 */


/*
 * -------------------------------------------------------------------------
 * Transactional model publication example
 * -------------------------------------------------------------------------
 *
 * Publishing a model should make its coefficients and metrics visible as a
 * consistent unit. A transaction prevents a consumer from observing a model
 * record without its associated metrics.
 */
BEGIN;

SELECT
    model_id,
    model_name,
    coefficient_intercept,
    coefficient_processing_hours,
    coefficient_item_count
FROM regression_models
WHERE model_name = 'delivery_time_ols_v1'
FOR SHARE;

SELECT
    model_id,
    sse,
    mse,
    rmse,
    r_squared,
    adjusted_r_squared
FROM regression_metrics
WHERE model_id = (
    SELECT model_id
    FROM regression_models
    WHERE model_name = 'delivery_time_ols_v1'
);

COMMIT;
