-- ============================================================
-- ACTIVATION FUNCTION DATABASE
-- PostgreSQL 14+
--
-- Models:
-- 1. Activation definitions
-- 2. Neural network models
-- 3. Ordered activation layers
-- 4. Activation configurations
-- 5. Inference evaluation runs
-- 6. Classification outputs
-- 7. Numerical and operational validation
--
-- Supported activations:
-- Sigmoid, Tanh, ReLU, Leaky ReLU, GELU, Softmax
-- ============================================================

CREATE SCHEMA IF NOT EXISTS activation_lab;

SET search_path TO activation_lab, public;

-- ------------------------------------------------------------
-- ACTIVATION CATALOG
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS activation_definitions (
    activation_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    activation_name VARCHAR(40) NOT NULL UNIQUE,

    activation_category VARCHAR(30) NOT NULL,

    output_range TEXT NOT NULL,

    differentiable BOOLEAN NOT NULL,

    vector_operation BOOLEAN NOT NULL DEFAULT FALSE,

    description TEXT NOT NULL,

    CHECK (
        activation_name IN (
            'sigmoid',
            'tanh',
            'relu',
            'leaky_relu',
            'gelu',
            'softmax'
        )
    ),

    CHECK (
        activation_category IN (
            'bounded',
            'unbounded',
            'probability'
        )
    )
);

INSERT INTO activation_definitions (
    activation_name,
    activation_category,
    output_range,
    differentiable,
    vector_operation,
    description
)
VALUES
(
    'sigmoid',
    'bounded',
    '(0,1)',
    TRUE,
    FALSE,
    'Logistic activation for binary probability outputs.'
),
(
    'tanh',
    'bounded',
    '(-1,1)',
    TRUE,
    FALSE,
    'Zero-centered hyperbolic tangent activation.'
),
(
    'relu',
    'unbounded',
    '[0,infinity)',
    TRUE,
    FALSE,
    'Rectified linear activation.'
),
(
    'leaky_relu',
    'unbounded',
    '(-infinity,infinity)',
    TRUE,
    FALSE,
    'Rectified linear activation with a nonzero negative slope.'
),
(
    'gelu',
    'unbounded',
    'Approximately (-0.17,infinity)',
    TRUE,
    FALSE,
    'Gaussian error linear activation using tanh approximation.'
),
(
    'softmax',
    'probability',
    '(0,1), sum equals 1',
    TRUE,
    TRUE,
    'Vector normalization for multiclass probabilities.'
)
ON CONFLICT (activation_name) DO NOTHING;


-- ------------------------------------------------------------
-- MODEL MANAGEMENT
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS neural_models (
    model_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    model_name VARCHAR(150) NOT NULL UNIQUE,

    model_version VARCHAR(30) NOT NULL,

    model_purpose VARCHAR(40) NOT NULL,

    input_dimensions INTEGER NOT NULL,

    output_dimensions INTEGER NOT NULL,

    model_status VARCHAR(20) NOT NULL DEFAULT 'draft',

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CHECK (input_dimensions > 0),

    CHECK (output_dimensions > 0),

    CHECK (
        model_status IN (
            'draft',
            'testing',
            'validated',
            'production',
            'retired'
        )
    ),

    CHECK (
        model_purpose IN (
            'classification',
            'regression',
            'representation',
            'multilabel'
        )
    )
);


-- ------------------------------------------------------------
-- MODEL LAYERS
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS model_layers (
    layer_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    model_id BIGINT NOT NULL
        REFERENCES neural_models(model_id)
        ON DELETE CASCADE,

    layer_name VARCHAR(100) NOT NULL,

    layer_position INTEGER NOT NULL,

    activation_id INTEGER NOT NULL
        REFERENCES activation_definitions(activation_id),

    input_dimensions INTEGER NOT NULL,

    output_dimensions INTEGER NOT NULL,

    negative_slope DOUBLE PRECISION NOT NULL DEFAULT 0.01,

    is_output_layer BOOLEAN NOT NULL DEFAULT FALSE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (model_id, layer_position),

    UNIQUE (model_id, layer_name),

    CHECK (layer_position >= 0),

    CHECK (input_dimensions > 0),

    CHECK (output_dimensions > 0),

    CHECK (
        negative_slope >= 0
        AND negative_slope < 'Infinity'::DOUBLE PRECISION
    )
);

CREATE INDEX IF NOT EXISTS idx_model_layers_activation
ON model_layers(activation_id, model_id);

CREATE INDEX IF NOT EXISTS idx_model_layers_model_position
ON model_layers(model_id, layer_position);


-- ------------------------------------------------------------
-- INFERENCE RUNS
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    model_id BIGINT NOT NULL
        REFERENCES neural_models(model_id),

    execution_environment VARCHAR(30) NOT NULL,

    execution_status VARCHAR(20) NOT NULL DEFAULT 'queued',

    started_at TIMESTAMPTZ,

    completed_at TIMESTAMPTZ,

    total_samples INTEGER NOT NULL DEFAULT 0,

    average_latency_ms DOUBLE PRECISION,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CHECK (
        execution_environment IN (
            'development',
            'staging',
            'production'
        )
    ),

    CHECK (
        execution_status IN (
            'queued',
            'running',
            'completed',
            'failed'
        )
    ),

    CHECK (total_samples >= 0),

    CHECK (
        average_latency_ms IS NULL
        OR (
            average_latency_ms >= 0
            AND average_latency_ms < 'Infinity'::DOUBLE PRECISION
        )
    ),

    CHECK (
        completed_at IS NULL
        OR started_at IS NULL
        OR completed_at >= started_at
    )
);

CREATE INDEX IF NOT EXISTS idx_evaluation_runs_model
ON evaluation_runs(model_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_evaluation_runs_status
ON evaluation_runs(execution_status);


-- ------------------------------------------------------------
-- SAMPLE-LEVEL OUTPUTS
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS inference_outputs (
    output_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    run_id BIGINT NOT NULL
        REFERENCES evaluation_runs(run_id)
        ON DELETE CASCADE,

    sample_reference VARCHAR(100) NOT NULL,

    raw_logits DOUBLE PRECISION[] NOT NULL,

    activation_outputs DOUBLE PRECISION[] NOT NULL,

    predicted_class INTEGER,

    actual_class INTEGER,

    loss_value DOUBLE PRECISION,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (run_id, sample_reference),

    CHECK (cardinality(raw_logits) > 0),

    CHECK (cardinality(activation_outputs) > 0),

    CHECK (
        predicted_class IS NULL
        OR predicted_class >= 0
    ),

    CHECK (
        actual_class IS NULL
        OR actual_class >= 0
    ),

    CHECK (
        loss_value IS NULL
        OR (
            loss_value >= 0
            AND loss_value < 'Infinity'::DOUBLE PRECISION
        )
    )
);

CREATE INDEX IF NOT EXISTS idx_inference_outputs_run
ON inference_outputs(run_id);

CREATE INDEX IF NOT EXISTS idx_inference_outputs_prediction
ON inference_outputs(predicted_class);


-- ------------------------------------------------------------
-- ACTIVATION FUNCTIONS
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION sigmoid(x DOUBLE PRECISION)
RETURNS DOUBLE PRECISION
LANGUAGE SQL
IMMUTABLE
STRICT
AS $$
    SELECT CASE
        WHEN x >= 0
            THEN 1.0 / (1.0 + EXP(-x))
        ELSE
            EXP(x) / (1.0 + EXP(x))
    END;
$$;


CREATE OR REPLACE FUNCTION tanh_activation(x DOUBLE PRECISION)
RETURNS DOUBLE PRECISION
LANGUAGE SQL
IMMUTABLE
STRICT
AS $$
    SELECT TANH(x);
$$;


CREATE OR REPLACE FUNCTION relu(x DOUBLE PRECISION)
RETURNS DOUBLE PRECISION
LANGUAGE SQL
IMMUTABLE
STRICT
AS $$
    SELECT GREATEST(0.0, x);
$$;


CREATE OR REPLACE FUNCTION leaky_relu(
    x DOUBLE PRECISION,
    alpha DOUBLE PRECISION DEFAULT 0.01
)
RETURNS DOUBLE PRECISION
LANGUAGE PLPGSQL
IMMUTABLE
AS $$
BEGIN
    IF x IS NULL OR alpha IS NULL
       OR x IN (
           'Infinity'::DOUBLE PRECISION,
           '-Infinity'::DOUBLE PRECISION,
           'NaN'::DOUBLE PRECISION
       )
       OR alpha < 0
       OR alpha IN (
           'Infinity'::DOUBLE PRECISION,
           'NaN'::DOUBLE PRECISION
       )
    THEN
        RAISE EXCEPTION 'Invalid Leaky ReLU parameters';
    END IF;

    RETURN CASE WHEN x > 0 THEN x ELSE alpha * x END;
END;
$$;


CREATE OR REPLACE FUNCTION gelu(x DOUBLE PRECISION)
RETURNS DOUBLE PRECISION
LANGUAGE PLPGSQL
IMMUTABLE
STRICT
AS $$
DECLARE
    inner_value DOUBLE PRECISION;
BEGIN
    IF x IN (
        'Infinity'::DOUBLE PRECISION,
        '-Infinity'::DOUBLE PRECISION,
        'NaN'::DOUBLE PRECISION
    ) THEN
        RAISE EXCEPTION 'GELU requires finite input';
    END IF;

    inner_value := SQRT(2.0 / PI()) *
        (x + 0.044715 * POWER(x, 3));

    RETURN 0.5 * x * (1.0 + TANH(inner_value));
END;
$$;


-- ------------------------------------------------------------
-- SOFTMAX
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION softmax(
    logits DOUBLE PRECISION[]
)
RETURNS DOUBLE PRECISION[]
LANGUAGE PLPGSQL
IMMUTABLE
AS $$
DECLARE
    maximum_value DOUBLE PRECISION;
    denominator DOUBLE PRECISION;
    result DOUBLE PRECISION[];
    value DOUBLE PRECISION;
BEGIN
    IF logits IS NULL OR cardinality(logits) = 0 THEN
        RAISE EXCEPTION 'Softmax input cannot be empty';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM unnest(logits) AS item(value)
        WHERE value IS NULL
           OR value IN (
               'Infinity'::DOUBLE PRECISION,
               '-Infinity'::DOUBLE PRECISION,
               'NaN'::DOUBLE PRECISION
           )
    ) THEN
        RAISE EXCEPTION 'Softmax requires finite inputs';
    END IF;

    SELECT MAX(item.value)
    INTO maximum_value
    FROM unnest(logits) AS item(value);

    SELECT SUM(EXP(item.value - maximum_value))
    INTO denominator
    FROM unnest(logits) AS item(value);

    IF denominator IS NULL OR denominator <= 0 THEN
        RAISE EXCEPTION 'Softmax normalization failed';
    END IF;

    SELECT ARRAY_AGG(
        EXP(item.value - maximum_value) / denominator
        ORDER BY item.ordinality
    )
    INTO result
    FROM unnest(logits) WITH ORDINALITY AS item(value, ordinality);

    RETURN result;
END;
$$;


-- ------------------------------------------------------------
-- DERIVATIVES
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION activation_derivative(
    activation_name VARCHAR,
    x DOUBLE PRECISION,
    alpha DOUBLE PRECISION DEFAULT 0.01
)
RETURNS DOUBLE PRECISION
LANGUAGE PLPGSQL
IMMUTABLE
AS $$
DECLARE
    y DOUBLE PRECISION;
    inner_value DOUBLE PRECISION;
    t DOUBLE PRECISION;
    c DOUBLE PRECISION;
BEGIN
    IF x IS NULL OR alpha IS NULL
       OR x IN (
           'Infinity'::DOUBLE PRECISION,
           '-Infinity'::DOUBLE PRECISION,
           'NaN'::DOUBLE PRECISION
       )
       OR alpha < 0
       OR alpha IN (
           'Infinity'::DOUBLE PRECISION,
           'NaN'::DOUBLE PRECISION
       )
    THEN
        RAISE EXCEPTION 'Invalid derivative input';
    END IF;

    CASE activation_name

        WHEN 'sigmoid' THEN
            y := sigmoid(x);
            RETURN y * (1.0 - y);

        WHEN 'tanh' THEN
            y := TANH(x);
            RETURN 1.0 - y * y;

        WHEN 'relu' THEN
            RETURN CASE WHEN x > 0 THEN 1.0 ELSE 0.0 END;

        WHEN 'leaky_relu' THEN
            RETURN CASE WHEN x > 0 THEN 1.0 ELSE alpha END;

        WHEN 'gelu' THEN
            c := SQRT(2.0 / PI());

            inner_value := c *
                (x + 0.044715 * POWER(x, 3));

            t := TANH(inner_value);

            RETURN 0.5 * (1.0 + t)
                + 0.5 * x * (1.0 - t * t)
                * c * (1.0 + 3.0 * 0.044715 * x * x);

        WHEN 'softmax' THEN
            RAISE EXCEPTION
                'Softmax requires vector-level gradient calculation';

        ELSE
            RAISE EXCEPTION
                'Unknown activation: %', activation_name;
    END CASE;
END;
$$;


-- ------------------------------------------------------------
-- SOFTMAX JACOBIAN-VECTOR PRODUCT
-- O(n) implementation instead of materializing O(n^2) matrix.
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION softmax_jvp(
    probabilities DOUBLE PRECISION[],
    upstream DOUBLE PRECISION[]
)
RETURNS DOUBLE PRECISION[]
LANGUAGE PLPGSQL
IMMUTABLE
AS $$
DECLARE
    dot_product DOUBLE PRECISION;
    total DOUBLE PRECISION;
    result DOUBLE PRECISION[];
BEGIN
    IF probabilities IS NULL
       OR upstream IS NULL
       OR cardinality(probabilities) = 0
       OR cardinality(probabilities) <> cardinality(upstream)
    THEN
        RAISE EXCEPTION 'Invalid softmax gradient dimensions';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM unnest(probabilities) AS p(value)
        WHERE value < 0 OR value IS NULL
    ) OR EXISTS (
        SELECT 1
        FROM unnest(upstream) AS g(value)
        WHERE value IS NULL
           OR value IN (
               'Infinity'::DOUBLE PRECISION,
               '-Infinity'::DOUBLE PRECISION,
               'NaN'::DOUBLE PRECISION
           )
    ) THEN
        RAISE EXCEPTION 'Invalid gradient or probability values';
    END IF;

    SELECT SUM(value)
    INTO total
    FROM unnest(probabilities) AS p(value);

    IF ABS(total - 1.0) > 0.0000001 THEN
        RAISE EXCEPTION 'Probabilities must sum to one';
    END IF;

    SELECT SUM(p.value * g.value)
    INTO dot_product
    FROM unnest(probabilities) WITH ORDINALITY
        AS p(value, position)
    JOIN unnest(upstream) WITH ORDINALITY
        AS g(value, position)
    USING (position);

    SELECT ARRAY_AGG(
        p.value * (g.value - dot_product)
        ORDER BY p.position
    )
    INTO result
    FROM unnest(probabilities) WITH ORDINALITY
        AS p(value, position)
    JOIN unnest(upstream) WITH ORDINALITY
        AS g(value, position)
    USING (position);

    RETURN result;
END;
$$;


-- ------------------------------------------------------------
-- MODEL LAYER VALIDATION
-- Softmax is restricted to output layers.
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION validate_model_layer()
RETURNS TRIGGER
LANGUAGE PLPGSQL
AS $$
DECLARE
    activation VARCHAR(40);
    purpose VARCHAR(40);
BEGIN
    SELECT a.activation_name, m.model_purpose
    INTO activation, purpose
    FROM activation_definitions a
    CROSS JOIN neural_models m
    WHERE a.activation_id = NEW.activation_id
      AND m.model_id = NEW.model_id;

    IF activation IS NULL THEN
        RAISE EXCEPTION 'Invalid activation or model';
    END IF;

    IF activation = 'softmax'
       AND NOT NEW.is_output_layer
    THEN
        RAISE EXCEPTION
            'Softmax is configured as an output layer only';
    END IF;

    IF NEW.is_output_layer
       AND purpose = 'classification'
       AND activation NOT IN ('softmax', 'sigmoid')
    THEN
        RAISE EXCEPTION
            'Classification output requires sigmoid or softmax';
    END IF;

    RETURN NEW;
END;
$$;


DROP TRIGGER IF EXISTS trg_validate_model_layer
ON model_layers;

CREATE TRIGGER trg_validate_model_layer
BEFORE INSERT OR UPDATE ON model_layers
FOR EACH ROW
EXECUTE FUNCTION validate_model_layer();


-- ------------------------------------------------------------
-- EVALUATION STATE TRANSITIONS
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION validate_evaluation_transition()
RETURNS TRIGGER
LANGUAGE PLPGSQL
AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN

        IF OLD.execution_status IN ('completed', 'failed')
           AND NEW.execution_status <> OLD.execution_status
        THEN
            RAISE EXCEPTION
                'Terminal evaluation status cannot be changed';
        END IF;

        IF OLD.execution_status = 'queued'
           AND NEW.execution_status NOT IN ('queued', 'running', 'failed')
        THEN
            RAISE EXCEPTION 'Invalid queued state transition';
        END IF;

        IF OLD.execution_status = 'running'
           AND NEW.execution_status NOT IN ('running', 'completed', 'failed')
        THEN
            RAISE EXCEPTION 'Invalid running state transition';
        END IF;

    END IF;

    IF NEW.execution_status = 'running'
       AND NEW.started_at IS NULL
    THEN
        NEW.started_at := NOW();
    END IF;

    IF NEW.execution_status IN ('completed', 'failed')
       AND NEW.completed_at IS NULL
    THEN
        NEW.completed_at := NOW();
    END IF;

    RETURN NEW;
END;
$$;


DROP TRIGGER IF EXISTS trg_evaluation_transition
ON evaluation_runs;

CREATE TRIGGER trg_evaluation_transition
BEFORE UPDATE ON evaluation_runs
FOR EACH ROW
EXECUTE FUNCTION validate_evaluation_transition();


-- ------------------------------------------------------------
-- MODEL ELIGIBILITY VIEW
-- ------------------------------------------------------------

CREATE OR REPLACE VIEW model_activation_report AS
SELECT
    m.model_id,
    m.model_name,
    m.model_version,
    m.model_purpose,
    m.model_status,

    COUNT(l.layer_id) AS total_layers,

    COUNT(*) FILTER (
        WHERE a.activation_name = 'relu'
    ) AS relu_layers,

    COUNT(*) FILTER (
        WHERE a.activation_name = 'gelu'
    ) AS gelu_layers,

    COUNT(*) FILTER (
        WHERE a.activation_name = 'softmax'
    ) AS softmax_layers,

    BOOL_OR(l.is_output_layer) AS has_output_layer

FROM neural_models m

LEFT JOIN model_layers l
    ON l.model_id = m.model_id

LEFT JOIN activation_definitions a
    ON a.activation_id = l.activation_id

GROUP BY
    m.model_id,
    m.model_name,
    m.model_version,
    m.model_purpose,
    m.model_status;


-- ------------------------------------------------------------
-- SOFTMAX CLASSIFICATION REPORT
-- ------------------------------------------------------------

CREATE OR REPLACE VIEW classification_accuracy_report AS
SELECT
    r.run_id,
    m.model_name,

    COUNT(o.output_id) AS total_predictions,

    COUNT(*) FILTER (
        WHERE o.predicted_class = o.actual_class
    ) AS correct_predictions,

    ROUND(
        100.0 *
        COUNT(*) FILTER (
            WHERE o.predicted_class = o.actual_class
        )
        / NULLIF(COUNT(o.output_id), 0),
        2
    ) AS accuracy_percentage,

    ROUND(AVG(o.loss_value)::NUMERIC, 6) AS average_loss

FROM evaluation_runs r

JOIN neural_models m
    ON m.model_id = r.model_id

LEFT JOIN inference_outputs o
    ON o.run_id = r.run_id

GROUP BY r.run_id, m.model_name;


-- ------------------------------------------------------------
-- STORED PROCEDURE: COMPLETE AN EVALUATION
-- ------------------------------------------------------------

CREATE OR REPLACE PROCEDURE complete_evaluation(
    p_run_id BIGINT,
    p_average_latency DOUBLE PRECISION
)
LANGUAGE PLPGSQL
AS $$
BEGIN
    IF p_average_latency IS NULL
       OR p_average_latency < 0
       OR p_average_latency IN (
           'Infinity'::DOUBLE PRECISION,
           'NaN'::DOUBLE PRECISION
       )
    THEN
        RAISE EXCEPTION 'Invalid latency value';
    END IF;

    UPDATE evaluation_runs
    SET
        execution_status = 'completed',
        completed_at = NOW(),
        average_latency_ms = p_average_latency

    WHERE run_id = p_run_id
      AND execution_status = 'running';

    IF NOT FOUND THEN
        RAISE EXCEPTION
            'Evaluation % is not in running state',
            p_run_id;
    END IF;
END;
$$;


-- ------------------------------------------------------------
-- SAMPLE MODEL
-- ------------------------------------------------------------

INSERT INTO neural_models (
    model_name,
    model_version,
    model_purpose,
    input_dimensions,
    output_dimensions,
    model_status
)
VALUES
(
    'Customer Risk Classifier',
    '1.0',
    'classification',
    4,
    3,
    'testing'
),
(
    'Regression Feature Network',
    '1.0',
    'regression',
    4,
    1,
    'draft'
)
ON CONFLICT (model_name) DO NOTHING;


-- ------------------------------------------------------------
-- SAMPLE LAYERS
-- ------------------------------------------------------------

INSERT INTO model_layers (
    model_id,
    layer_name,
    layer_position,
    activation_id,
    input_dimensions,
    output_dimensions,
    is_output_layer
)
SELECT
    m.model_id,
    v.layer_name,
    v.layer_position,
    a.activation_id,
    v.input_dimensions,
    v.output_dimensions,
    v.is_output_layer

FROM (
    VALUES
        ('Customer Risk Classifier', 'hidden_relu', 1, 'relu', 4, 8, FALSE),
        ('Customer Risk Classifier', 'hidden_gelu', 2, 'gelu', 8, 6, FALSE),
        ('Customer Risk Classifier', 'output_softmax', 3, 'softmax', 6, 3, TRUE),
        ('Regression Feature Network', 'regression_tanh', 1, 'tanh', 4, 4, FALSE)
) AS v(
    model_name,
    layer_name,
    layer_position,
    activation_name,
    input_dimensions,
    output_dimensions,
    is_output_layer
)

JOIN neural_models m
    ON m.model_name = v.model_name

JOIN activation_definitions a
    ON a.activation_name = v.activation_name

ON CONFLICT (model_id, layer_position) DO NOTHING;


-- ------------------------------------------------------------
-- EVALUATION EXECUTION RECORDS
-- ------------------------------------------------------------

INSERT INTO evaluation_runs (
    model_id,
    execution_environment,
    execution_status,
    started_at
)
SELECT
    model_id,
    'staging',
    'running',
    NOW()
FROM neural_models
WHERE model_name = 'Customer Risk Classifier'
AND NOT EXISTS (
    SELECT 1
    FROM evaluation_runs r
    WHERE r.model_id = neural_models.model_id
);


-- ------------------------------------------------------------
-- SAMPLE INFERENCE RESULTS
-- ------------------------------------------------------------

INSERT INTO inference_outputs (
    run_id,
    sample_reference,
    raw_logits,
    activation_outputs,
    predicted_class,
    actual_class,
    loss_value
)
SELECT
    r.run_id,
    sample.sample_reference,
    sample.logits,
    softmax(sample.logits),
    sample.predicted_class,
    sample.actual_class,
    -LN(
        GREATEST(
            0.000000000000001,
            softmax(sample.logits)[sample.actual_class + 1]
        )
    )

FROM evaluation_runs r

JOIN neural_models m
    ON m.model_id = r.model_id

CROSS JOIN (
    VALUES
        (
            'CUSTOMER-001',
            ARRAY[2.0, 1.0, 0.1]::DOUBLE PRECISION[],
            0,
            0
        ),
        (
            'CUSTOMER-002',
            ARRAY[0.2, 2.5, 0.4]::DOUBLE PRECISION[],
            1,
            1
        ),
        (
            'CUSTOMER-003',
            ARRAY[0.1, 0.3, 2.2]::DOUBLE PRECISION[],
            2,
            1
        )
) AS sample(
    sample_reference,
    logits,
    predicted_class,
    actual_class
)

WHERE m.model_name = 'Customer Risk Classifier'

ON CONFLICT (run_id, sample_reference) DO NOTHING;


-- ------------------------------------------------------------
-- TRANSACTIONAL WORKFLOW
-- ------------------------------------------------------------

BEGIN;

UPDATE evaluation_runs r
SET
    total_samples = (
        SELECT COUNT(*)
        FROM inference_outputs o
        WHERE o.run_id = r.run_id
    )
WHERE r.execution_status = 'running';

COMMIT;


-- ------------------------------------------------------------
-- NUMERICAL FUNCTION RETRIEVAL
-- ------------------------------------------------------------

SELECT
    value AS input,
    sigmoid(value) AS sigmoid_output,
    tanh_activation(value) AS tanh_output,
    relu(value) AS relu_output,
    leaky_relu(value, 0.01) AS leaky_relu_output,
    gelu(value) AS gelu_output

FROM unnest(
    ARRAY[-3.0, -1.0, 0.0, 1.0, 3.0]::DOUBLE PRECISION[]
) AS inputs(value);


-- ------------------------------------------------------------
-- DERIVATIVE EVALUATION
-- ------------------------------------------------------------

SELECT
    value AS input,

    activation_derivative('sigmoid', value)
        AS sigmoid_gradient,

    activation_derivative('tanh', value)
        AS tanh_gradient,

    activation_derivative('relu', value)
        AS relu_gradient,

    activation_derivative('leaky_relu', value, 0.01)
        AS leaky_relu_gradient,

    activation_derivative('gelu', value)
        AS gelu_gradient

FROM unnest(
    ARRAY[-2.0, -1.0, 0.0, 1.0, 2.0]::DOUBLE PRECISION[]
) AS inputs(value);


-- ------------------------------------------------------------
-- SOFTMAX EVALUATION
-- ------------------------------------------------------------

SELECT
    ARRAY[2.0, 1.0, 0.1]::DOUBLE PRECISION[] AS logits,

    softmax(
        ARRAY[2.0, 1.0, 0.1]::DOUBLE PRECISION[]
    ) AS probabilities;


-- ------------------------------------------------------------
-- SOFTMAX GRADIENT
-- ------------------------------------------------------------

SELECT softmax_jvp(
    softmax(ARRAY[2.0, 1.0, 0.1]::DOUBLE PRECISION[]),
    ARRAY[1.0, 0.0, -1.0]::DOUBLE PRECISION[]
) AS gradient;


-- ------------------------------------------------------------
-- MODEL CONFIGURATION RETRIEVAL
-- ------------------------------------------------------------

SELECT
    m.model_name,
    l.layer_position,
    l.layer_name,
    a.activation_name,
    l.input_dimensions,
    l.output_dimensions,
    l.is_output_layer

FROM neural_models m

JOIN model_layers l
    ON l.model_id = m.model_id

JOIN activation_definitions a
    ON a.activation_id = l.activation_id

ORDER BY m.model_name, l.layer_position;


-- ------------------------------------------------------------
-- MODEL VALIDATION
-- ------------------------------------------------------------

SELECT
    m.model_name,

    CASE
        WHEN COUNT(l.layer_id) = 0
            THEN 'INVALID: No activation layers'

        WHEN NOT BOOL_OR(l.is_output_layer)
            THEN 'INVALID: Missing output layer'

        WHEN m.model_status = 'production'
            THEN 'PRODUCTION'

        WHEN m.model_status = 'validated'
            THEN 'VALIDATED'

        ELSE 'REQUIRES VALIDATION'
    END AS validation_status

FROM neural_models m

LEFT JOIN model_layers l
    ON l.model_id = m.model_id

GROUP BY m.model_id, m.model_name, m.model_status;


-- ------------------------------------------------------------
-- ACTIVATION USAGE AGGREGATION
-- ------------------------------------------------------------

SELECT
    a.activation_name,
    a.activation_category,
    COUNT(l.layer_id) AS usage_count,
    COUNT(DISTINCT l.model_id) AS model_count

FROM activation_definitions a

LEFT JOIN model_layers l
    ON l.activation_id = a.activation_id

GROUP BY
    a.activation_id,
    a.activation_name,
    a.activation_category

ORDER BY usage_count DESC, a.activation_name;


-- ------------------------------------------------------------
-- EVALUATION REPORTING
-- ------------------------------------------------------------

SELECT
    run_id,
    model_name,
    total_predictions,
    correct_predictions,
    accuracy_percentage,
    average_loss

FROM classification_accuracy_report

ORDER BY run_id;


-- ------------------------------------------------------------
-- EXECUTION PERFORMANCE REPORT
-- ------------------------------------------------------------

SELECT
    m.model_name,
    r.execution_environment,
    r.execution_status,
    r.total_samples,
    r.average_latency_ms,
    r.created_at

FROM evaluation_runs r

JOIN neural_models m
    ON m.model_id = r.model_id

ORDER BY r.created_at DESC;


-- ------------------------------------------------------------
-- FAILURE DIAGNOSTICS
-- ------------------------------------------------------------

SELECT
    r.run_id,
    m.model_name,
    r.execution_status,
    r.started_at,
    r.completed_at

FROM evaluation_runs r

JOIN neural_models m
    ON m.model_id = r.model_id

WHERE r.execution_status = 'failed';


-- ------------------------------------------------------------
-- COMPLETION EXAMPLE
-- ------------------------------------------------------------

-- Execute only for a run currently marked as running.
-- CALL complete_evaluation(1, 2.5);

-- The following queries inspect the final database state.

SELECT * FROM model_activation_report;

SELECT * FROM classification_accuracy_report;
