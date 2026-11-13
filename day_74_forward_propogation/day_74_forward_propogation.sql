-- ============================================================
-- FORWARD PROPAGATION IN NEURAL NETWORKS
-- PostgreSQL 14+
--
-- Models:
--   Neural network architectures
--   Dense layers
--   Neurons
--   Weights
--   Biases
--   Activation functions
--   Inference requests
--   Layer execution results
--   Model validation
--   Prediction reporting
--
-- Matrix orientation:
--   weights[output_neuron][input_neuron]
--
-- Forward computation:
--   Z = W * A + B
--   A_next = activation(Z)
-- ============================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS neural_network;

SET search_path TO neural_network, public;

-- ------------------------------------------------------------
-- 1. DOMAIN TYPES
-- ------------------------------------------------------------

CREATE TYPE activation_type AS ENUM (
    'linear',
    'relu',
    'sigmoid',
    'tanh',
    'softmax'
);

CREATE TYPE model_status AS ENUM (
    'draft',
    'validated',
    'active',
    'archived'
);

CREATE TYPE inference_status AS ENUM (
    'pending',
    'running',
    'completed',
    'failed'
);

-- ------------------------------------------------------------
-- 2. MODEL DEFINITIONS
-- ------------------------------------------------------------

CREATE TABLE neural_models (
    model_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    model_name VARCHAR(150) NOT NULL UNIQUE,

    description TEXT,

    input_features INTEGER NOT NULL
        CHECK (input_features > 0),

    output_features INTEGER NOT NULL
        CHECK (output_features > 0),

    model_version INTEGER NOT NULL DEFAULT 1
        CHECK (model_version > 0),

    status model_status NOT NULL DEFAULT 'draft',

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE dense_layers (
    layer_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    model_id BIGINT NOT NULL
        REFERENCES neural_models(model_id)
        ON DELETE CASCADE,

    layer_index INTEGER NOT NULL
        CHECK (layer_index > 0),

    layer_name VARCHAR(100) NOT NULL,

    input_size INTEGER NOT NULL
        CHECK (input_size > 0),

    output_size INTEGER NOT NULL
        CHECK (output_size > 0),

    activation activation_type NOT NULL,

    UNIQUE (model_id, layer_index),

    UNIQUE (model_id, layer_name)
);

-- Weights are stored as JSON arrays of output-neuron rows.
-- Biases contain one value per output neuron.
CREATE TABLE layer_parameters (
    layer_id BIGINT PRIMARY KEY
        REFERENCES dense_layers(layer_id)
        ON DELETE CASCADE,

    weights JSONB NOT NULL,

    biases DOUBLE PRECISION[] NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CHECK (jsonb_typeof(weights) = 'array'),

    CHECK (jsonb_array_length(weights) > 0),

    CHECK (cardinality(biases) > 0)
);

-- ------------------------------------------------------------
-- 3. INFERENCE REQUESTS
-- ------------------------------------------------------------

CREATE TABLE inference_requests (
    inference_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    model_id BIGINT NOT NULL
        REFERENCES neural_models(model_id),

    input_vector DOUBLE PRECISION[] NOT NULL,

    status inference_status NOT NULL DEFAULT 'pending',

    output_vector DOUBLE PRECISION[],

    error_message TEXT,

    requested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    completed_at TIMESTAMPTZ,

    CHECK (cardinality(input_vector) > 0),

    CHECK (
        output_vector IS NULL
        OR cardinality(output_vector) > 0
    )
);

CREATE TABLE layer_execution_results (
    execution_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    inference_id BIGINT NOT NULL
        REFERENCES inference_requests(inference_id)
        ON DELETE CASCADE,

    layer_id BIGINT NOT NULL
        REFERENCES dense_layers(layer_id),

    weighted_sums DOUBLE PRECISION[] NOT NULL,

    activated_outputs DOUBLE PRECISION[] NOT NULL,

    executed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (inference_id, layer_id)
);

-- ------------------------------------------------------------
-- 4. PERFORMANCE INDEXES
-- ------------------------------------------------------------

CREATE INDEX idx_dense_layers_model_order
    ON dense_layers(model_id, layer_index);

CREATE INDEX idx_inference_model_time
    ON inference_requests(model_id, requested_at DESC);

CREATE INDEX idx_inference_status_time
    ON inference_requests(status, requested_at);

CREATE INDEX idx_execution_inference
    ON layer_execution_results(inference_id);

-- ------------------------------------------------------------
-- 5. INPUT VALIDATION FUNCTION
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION validate_numeric_vector(
    p_values DOUBLE PRECISION[]
)
RETURNS BOOLEAN
LANGUAGE plpgsql
IMMUTABLE
STRICT
AS $$
DECLARE
    v_value DOUBLE PRECISION;
BEGIN
    IF cardinality(p_values) = 0 THEN
        RETURN FALSE;
    END IF;

    FOREACH v_value IN ARRAY p_values
    LOOP
        IF v_value IS NULL
           OR v_value = 'NaN'::DOUBLE PRECISION
           OR v_value = 'Infinity'::DOUBLE PRECISION
           OR v_value = '-Infinity'::DOUBLE PRECISION THEN
            RETURN FALSE;
        END IF;
    END LOOP;

    RETURN TRUE;
END;
$$;

-- ------------------------------------------------------------
-- 6. ACTIVATION FUNCTION
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION apply_activation(
    p_values DOUBLE PRECISION[],
    p_activation activation_type
)
RETURNS DOUBLE PRECISION[]
LANGUAGE plpgsql
IMMUTABLE
STRICT
AS $$
DECLARE
    v_result DOUBLE PRECISION[] := ARRAY[]::DOUBLE PRECISION[];
    v_value DOUBLE PRECISION;
    v_maximum DOUBLE PRECISION;
    v_total DOUBLE PRECISION := 0;
    v_exponential DOUBLE PRECISION;
    v_index INTEGER;
BEGIN
    IF NOT validate_numeric_vector(p_values) THEN
        RAISE EXCEPTION 'Invalid activation input.';
    END IF;

    -- Softmax uses maximum subtraction for numerical stability.
    IF p_activation = 'softmax' THEN

        SELECT MAX(value)
        INTO v_maximum
        FROM unnest(p_values) AS t(value);

        FOREACH v_value IN ARRAY p_values
        LOOP
            v_exponential := EXP(v_value - v_maximum);

            v_result := array_append(
                v_result,
                v_exponential
            );

            v_total := v_total + v_exponential;
        END LOOP;

        IF v_total <= 0 OR NOT isfinite(v_total) THEN
            RAISE EXCEPTION 'Invalid softmax denominator.';
        END IF;

        FOR v_index IN 1..cardinality(v_result)
        LOOP
            v_result[v_index] :=
                v_result[v_index] / v_total;
        END LOOP;

        RETURN v_result;
    END IF;

    FOREACH v_value IN ARRAY p_values
    LOOP
        CASE p_activation

            WHEN 'linear' THEN
                v_exponential := v_value;

            WHEN 'relu' THEN
                v_exponential := GREATEST(0, v_value);

            WHEN 'sigmoid' THEN
                IF v_value >= 0 THEN
                    v_exponential :=
                        1.0 / (1.0 + EXP(-v_value));
                ELSE
                    v_exponential :=
                        EXP(v_value) / (1.0 + EXP(v_value));
                END IF;

            WHEN 'tanh' THEN
                v_exponential := TANH(v_value);

            ELSE
                RAISE EXCEPTION 'Unsupported activation.';
        END CASE;

        IF NOT isfinite(v_exponential) THEN
            RAISE EXCEPTION 'Activation produced invalid output.';
        END IF;

        v_result := array_append(
            v_result,
            v_exponential
        );
    END LOOP;

    RETURN v_result;
END;
$$;

-- ------------------------------------------------------------
-- 7. LAYER PARAMETER VALIDATION
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION validate_layer_parameters(
    p_layer_id BIGINT
)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    v_layer dense_layers%ROWTYPE;
    v_weights JSONB;
    v_biases DOUBLE PRECISION[];
    v_row JSONB;
    v_value JSONB;
    v_row_index INTEGER;
    v_column_index INTEGER;
    v_count INTEGER;
    v_number DOUBLE PRECISION;
BEGIN
    SELECT *
    INTO v_layer
    FROM dense_layers
    WHERE layer_id = p_layer_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Layer % does not exist.', p_layer_id;
    END IF;

    SELECT weights, biases
    INTO v_weights, v_biases
    FROM layer_parameters
    WHERE layer_id = p_layer_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Parameters missing for layer %.',
            p_layer_id;
    END IF;

    IF jsonb_array_length(v_weights) <> v_layer.output_size THEN
        RAISE EXCEPTION
            'Weight row count does not match output size.';
    END IF;

    IF cardinality(v_biases) <> v_layer.output_size THEN
        RAISE EXCEPTION
            'Bias count does not match output size.';
    END IF;

    IF NOT validate_numeric_vector(v_biases) THEN
        RAISE EXCEPTION 'Invalid bias values.';
    END IF;

    FOR v_row_index IN
        0..jsonb_array_length(v_weights) - 1
    LOOP
        v_row := v_weights -> v_row_index;

        IF jsonb_typeof(v_row) <> 'array' THEN
            RAISE EXCEPTION 'Weight row must be an array.';
        END IF;

        IF jsonb_array_length(v_row) <> v_layer.input_size THEN
            RAISE EXCEPTION
                'Weight column count does not match input size.';
        END IF;

        FOR v_column_index IN
            0..jsonb_array_length(v_row) - 1
        LOOP
            v_value := v_row -> v_column_index;

            IF jsonb_typeof(v_value) <> 'number' THEN
                RAISE EXCEPTION 'Weights must be numeric.';
            END IF;

            v_number := (v_value #>> '{}')::DOUBLE PRECISION;

            IF NOT isfinite(v_number) THEN
                RAISE EXCEPTION 'Non-finite weight detected.';
            END IF;
        END LOOP;
    END LOOP;

    RETURN TRUE;
END;
$$;

-- ------------------------------------------------------------
-- 8. COMPLETE MODEL VALIDATION
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION validate_model(
    p_model_id BIGINT
)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    v_model neural_models%ROWTYPE;
    v_layer dense_layers%ROWTYPE;
    v_previous_output INTEGER;
    v_expected_input INTEGER;
    v_layer_count INTEGER;
BEGIN
    SELECT *
    INTO v_model
    FROM neural_models
    WHERE model_id = p_model_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Model % does not exist.', p_model_id;
    END IF;

    SELECT COUNT(*)
    INTO v_layer_count
    FROM dense_layers
    WHERE model_id = p_model_id;

    IF v_layer_count = 0 THEN
        RAISE EXCEPTION 'Model must contain at least one layer.';
    END IF;

    SELECT input_size
    INTO v_expected_input
    FROM dense_layers
    WHERE model_id = p_model_id
      AND layer_index = 1;

    IF v_expected_input IS DISTINCT FROM v_model.input_features THEN
        RAISE EXCEPTION
            'First layer input size does not match model input features.';
    END IF;

    v_previous_output := v_expected_input;

    FOR v_layer IN
        SELECT *
        FROM dense_layers
        WHERE model_id = p_model_id
        ORDER BY layer_index
    LOOP
        IF v_layer.input_size <> v_previous_output THEN
            RAISE EXCEPTION
                'Layer % has incompatible input dimensions.',
                v_layer.layer_index;
        END IF;

        PERFORM validate_layer_parameters(v_layer.layer_id);

        v_previous_output := v_layer.output_size;
    END LOOP;

    IF v_previous_output <> v_model.output_features THEN
        RAISE EXCEPTION
            'Final layer output does not match model output features.';
    END IF;

    RETURN TRUE;
END;
$$;

-- ------------------------------------------------------------
-- 9. MODEL STATE MANAGEMENT
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION protect_model_activation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.status = 'active'
       AND OLD.status IS DISTINCT FROM 'active' THEN

        PERFORM validate_model(NEW.model_id);

    END IF;

    NEW.updated_at := CURRENT_TIMESTAMP;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_model_activation
BEFORE UPDATE OF status
ON neural_models
FOR EACH ROW
EXECUTE FUNCTION protect_model_activation();

-- ------------------------------------------------------------
-- 10. MODEL ACTIVATION
-- ------------------------------------------------------------

CREATE OR REPLACE PROCEDURE activate_model(
    p_model_id BIGINT
)
LANGUAGE plpgsql
AS $$
BEGIN
    PERFORM validate_model(p_model_id);

    UPDATE neural_models
    SET status = 'active',
        updated_at = CURRENT_TIMESTAMP
    WHERE model_id = p_model_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Model does not exist.';
    END IF;
END;
$$;

-- ------------------------------------------------------------
-- 11. FORWARD PROPAGATION FUNCTION
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION forward_propagation(
    p_model_id BIGINT,
    p_input DOUBLE PRECISION[]
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_model neural_models%ROWTYPE;
    v_layer dense_layers%ROWTYPE;

    v_current DOUBLE PRECISION[];
    v_next DOUBLE PRECISION[];
    v_weighted_sums DOUBLE PRECISION[];

    v_weights JSONB;
    v_biases DOUBLE PRECISION[];

    v_row JSONB;
    v_weight DOUBLE PRECISION;
    v_sum DOUBLE PRECISION;

    v_output_index INTEGER;
    v_input_index INTEGER;

    v_layer_trace JSONB := '[]'::JSONB;
    v_layer_result JSONB;
BEGIN
    SELECT *
    INTO v_model
    FROM neural_models
    WHERE model_id = p_model_id
      AND status = 'active';

    IF NOT FOUND THEN
        RAISE EXCEPTION
            'Model does not exist or is not active.';
    END IF;

    IF NOT validate_numeric_vector(p_input) THEN
        RAISE EXCEPTION 'Invalid input vector.';
    END IF;

    IF cardinality(p_input) <> v_model.input_features THEN
        RAISE EXCEPTION
            'Expected % input features, received %.',
            v_model.input_features,
            cardinality(p_input);
    END IF;

    PERFORM validate_model(p_model_id);

    v_current := p_input;

    FOR v_layer IN
        SELECT *
        FROM dense_layers
        WHERE model_id = p_model_id
        ORDER BY layer_index
    LOOP
        SELECT weights, biases
        INTO v_weights, v_biases
        FROM layer_parameters
        WHERE layer_id = v_layer.layer_id;

        IF cardinality(v_current) <> v_layer.input_size THEN
            RAISE EXCEPTION
                'Runtime dimension mismatch in layer %.',
                v_layer.layer_name;
        END IF;

        v_weighted_sums := ARRAY[]::DOUBLE PRECISION[];

        FOR v_output_index IN 0..v_layer.output_size - 1
        LOOP
            v_row := v_weights -> v_output_index;

            v_sum := v_biases[v_output_index + 1];

            FOR v_input_index IN 0..v_layer.input_size - 1
            LOOP
                v_weight := (
                    v_row ->> v_input_index
                )::DOUBLE PRECISION;

                v_sum := v_sum +
                    v_weight * v_current[v_input_index + 1];
            END LOOP;

            IF NOT isfinite(v_sum) THEN
                RAISE EXCEPTION
                    'Non-finite weighted sum in layer %.',
                    v_layer.layer_name;
            END IF;

            v_weighted_sums := array_append(
                v_weighted_sums,
                v_sum
            );
        END LOOP;

        v_next := apply_activation(
            v_weighted_sums,
            v_layer.activation
        );

        v_layer_result := jsonb_build_object(
            'layer_index', v_layer.layer_index,
            'layer_name', v_layer.layer_name,
            'activation', v_layer.activation,
            'weighted_sums', to_jsonb(v_weighted_sums),
            'outputs', to_jsonb(v_next)
        );

        v_layer_trace := v_layer_trace || jsonb_build_array(
            v_layer_result
        );

        v_current := v_next;
    END LOOP;

    RETURN jsonb_build_object(
        'model_id', p_model_id,
        'input', to_jsonb(p_input),
        'output', to_jsonb(v_current),
        'layers', v_layer_trace,
        'status', 'completed'
    );
END;
$$;

-- ------------------------------------------------------------
-- 12. INFERENCE PROCESSING PROCEDURE
-- ------------------------------------------------------------

CREATE OR REPLACE PROCEDURE process_inference(
    p_inference_id BIGINT
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_request inference_requests%ROWTYPE;
    v_result JSONB;
    v_layer JSONB;
    v_layer_id BIGINT;
    v_layer_index INTEGER;
    v_weighted_sums DOUBLE PRECISION[];
    v_outputs DOUBLE PRECISION[];
BEGIN
    SELECT *
    INTO v_request
    FROM inference_requests
    WHERE inference_id = p_inference_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Inference request not found.';
    END IF;

    IF v_request.status <> 'pending' THEN
        RAISE EXCEPTION
            'Only pending requests can be processed.';
    END IF;

    UPDATE inference_requests
    SET status = 'running'
    WHERE inference_id = p_inference_id;

    BEGIN
        v_result := forward_propagation(
            v_request.model_id,
            v_request.input_vector
        );

        FOR v_layer IN
            SELECT value
            FROM jsonb_array_elements(v_result -> 'layers')
        LOOP
            v_layer_index := (
                v_layer ->> 'layer_index'
            )::INTEGER;

            SELECT layer_id
            INTO v_layer_id
            FROM dense_layers
            WHERE model_id = v_request.model_id
              AND layer_index = v_layer_index;

            SELECT ARRAY(
                SELECT jsonb_array_elements_text(
                    v_layer -> 'weighted_sums'
                )::DOUBLE PRECISION
            )
            INTO v_weighted_sums;

            SELECT ARRAY(
                SELECT jsonb_array_elements_text(
                    v_layer -> 'outputs'
                )::DOUBLE PRECISION
            )
            INTO v_outputs;

            INSERT INTO layer_execution_results (
                inference_id,
                layer_id,
                weighted_sums,
                activated_outputs
            )
            VALUES (
                p_inference_id,
                v_layer_id,
                v_weighted_sums,
                v_outputs
            );
        END LOOP;

        SELECT ARRAY(
            SELECT jsonb_array_elements_text(
                v_result -> 'output'
            )::DOUBLE PRECISION
        )
        INTO v_outputs;

        UPDATE inference_requests
        SET status = 'completed',
            output_vector = v_outputs,
            completed_at = CURRENT_TIMESTAMP,
            error_message = NULL
        WHERE inference_id = p_inference_id;

    EXCEPTION WHEN OTHERS THEN
        UPDATE inference_requests
        SET status = 'failed',
            error_message = SQLERRM,
            completed_at = CURRENT_TIMESTAMP
        WHERE inference_id = p_inference_id;

        -- The request remains recorded as failed.
        -- Re-raise to allow the caller to observe the failure.
        RAISE;
    END;
END;
$$;

-- ------------------------------------------------------------
-- 13. REPORTING VIEW
-- ------------------------------------------------------------

CREATE OR REPLACE VIEW inference_report AS
SELECT
    ir.inference_id,
    nm.model_name,
    nm.model_version,
    ir.status,
    ir.input_vector,
    ir.output_vector,
    ir.error_message,
    ir.requested_at,
    ir.completed_at,
    EXTRACT(
        EPOCH FROM (ir.completed_at - ir.requested_at)
    ) AS elapsed_seconds
FROM inference_requests ir
JOIN neural_models nm
    ON nm.model_id = ir.model_id;

-- ------------------------------------------------------------
-- 14. LAYER ARCHITECTURE VIEW
-- ------------------------------------------------------------

CREATE OR REPLACE VIEW model_architecture AS
SELECT
    nm.model_id,
    nm.model_name,
    nm.status,
    dl.layer_index,
    dl.layer_name,
    dl.input_size,
    dl.output_size,
    dl.activation,
    (
        dl.input_size * dl.output_size + dl.output_size
    ) AS parameter_count
FROM neural_models nm
JOIN dense_layers dl
    ON dl.model_id = nm.model_id;

-- ------------------------------------------------------------
-- 15. SAMPLE MODEL
-- ------------------------------------------------------------

INSERT INTO neural_models (
    model_name,
    description,
    input_features,
    output_features,
    status
)
VALUES (
    'customer_classification_network',
    'Two-layer dense neural network for binary classification.',
    3,
    2,
    'draft'
);

-- Hidden layer: 3 input features, 4 neurons.
INSERT INTO dense_layers (
    model_id,
    layer_index,
    layer_name,
    input_size,
    output_size,
    activation
)
SELECT
    model_id,
    1,
    'hidden_layer',
    3,
    4,
    'relu'
FROM neural_models
WHERE model_name = 'customer_classification_network';

-- Output layer: 4 hidden neurons, 2 output neurons.
INSERT INTO dense_layers (
    model_id,
    layer_index,
    layer_name,
    input_size,
    output_size,
    activation
)
SELECT
    model_id,
    2,
    'output_layer',
    4,
    2,
    'softmax'
FROM neural_models
WHERE model_name = 'customer_classification_network';

-- Hidden layer parameters.
INSERT INTO layer_parameters (
    layer_id,
    weights,
    biases
)
SELECT
    layer_id,
    '[
        [0.2, 0.4, -0.3],
        [0.5, -0.1, 0.2],
        [-0.2, 0.3, 0.6],
        [0.1, 0.2, 0.4]
    ]'::JSONB,
    ARRAY[0.1, -0.1, 0.05, 0.0]::DOUBLE PRECISION[]
FROM dense_layers
WHERE layer_name = 'hidden_layer';

-- Output layer parameters.
INSERT INTO layer_parameters (
    layer_id,
    weights,
    biases
)
SELECT
    layer_id,
    '[
        [0.3, -0.2, 0.4, 0.1],
        [-0.1, 0.5, 0.2, -0.3]
    ]'::JSONB,
    ARRAY[0.05, -0.05]::DOUBLE PRECISION[]
FROM dense_layers
WHERE layer_name = 'output_layer';

-- ------------------------------------------------------------
-- 16. INVALID MODEL EXAMPLE
-- ------------------------------------------------------------

INSERT INTO neural_models (
    model_name,
    description,
    input_features,
    output_features,
    status
)
VALUES (
    'invalid_dimension_example',
    'Draft model demonstrating architecture validation.',
    3,
    2,
    'draft'
);

INSERT INTO dense_layers (
    model_id,
    layer_index,
    layer_name,
    input_size,
    output_size,
    activation
)
SELECT
    model_id,
    1,
    'invalid_first_layer',
    2,
    4,
    'relu'
FROM neural_models
WHERE model_name = 'invalid_dimension_example';

INSERT INTO layer_parameters (
    layer_id,
    weights,
    biases
)
SELECT
    layer_id,
    '[
        [0.1, 0.2],
        [0.2, 0.3],
        [0.3, 0.4],
        [0.4, 0.5]
    ]'::JSONB,
    ARRAY[0.0, 0.0, 0.0, 0.0]::DOUBLE PRECISION[]
FROM dense_layers
WHERE layer_name = 'invalid_first_layer';

-- ------------------------------------------------------------
-- 17. ACTIVATE VALID MODEL
-- ------------------------------------------------------------

CALL activate_model(
    (
        SELECT model_id
        FROM neural_models
        WHERE model_name = 'customer_classification_network'
    )
);

-- ------------------------------------------------------------
-- 18. INSERT INFERENCE REQUESTS
-- ------------------------------------------------------------

INSERT INTO inference_requests (
    model_id,
    input_vector,
    status
)
SELECT
    model_id,
    ARRAY[0.2, 0.5, 0.8]::DOUBLE PRECISION[],
    'pending'
FROM neural_models
WHERE model_name = 'customer_classification_network';

INSERT INTO inference_requests (
    model_id,
    input_vector,
    status
)
SELECT
    model_id,
    ARRAY[0.7, 0.1, 0.4]::DOUBLE PRECISION[],
    'pending'
FROM neural_models
WHERE model_name = 'customer_classification_network';

-- ------------------------------------------------------------
-- 19. PROCESS INFERENCE TRANSACTION
-- ------------------------------------------------------------

DO $$
DECLARE
    v_inference_id BIGINT;
BEGIN
    FOR v_inference_id IN
        SELECT inference_id
        FROM neural_network.inference_requests
        WHERE status = 'pending'
        ORDER BY inference_id
    LOOP
        CALL neural_network.process_inference(
            v_inference_id
        );
    END LOOP;
END;
$$;

-- ------------------------------------------------------------
-- 20. RETRIEVAL QUERIES
-- ------------------------------------------------------------

-- Retrieve model architecture.
SELECT *
FROM model_architecture
ORDER BY model_id, layer_index;

-- Retrieve predictions.
SELECT *
FROM inference_report
ORDER BY inference_id;

-- Retrieve intermediate layer results.
SELECT
    ir.inference_id,
    dl.layer_name,
    ler.weighted_sums,
    ler.activated_outputs
FROM layer_execution_results ler
JOIN inference_requests ir
    ON ir.inference_id = ler.inference_id
JOIN dense_layers dl
    ON dl.layer_id = ler.layer_id
ORDER BY ir.inference_id, dl.layer_index;

-- ------------------------------------------------------------
-- 21. DIRECT FORWARD PROPAGATION QUERY
-- ------------------------------------------------------------

SELECT forward_propagation(
    (
        SELECT model_id
        FROM neural_models
        WHERE model_name = 'customer_classification_network'
    ),
    ARRAY[0.2, 0.5, 0.8]::DOUBLE PRECISION[]
) AS forward_result;

-- ------------------------------------------------------------
-- 22. VALIDATION QUERY
-- ------------------------------------------------------------

SELECT
    model_name,
    validate_model(model_id) AS architecture_valid
FROM neural_models
WHERE status = 'active';

-- ------------------------------------------------------------
-- 23. AGGREGATION AND REPORTING
-- ------------------------------------------------------------

SELECT
    nm.model_name,
    COUNT(ir.inference_id) AS total_requests,
    COUNT(*) FILTER (
        WHERE ir.status = 'completed'
    ) AS completed_requests,
    COUNT(*) FILTER (
        WHERE ir.status = 'failed'
    ) AS failed_requests,
    AVG(
        EXTRACT(
            EPOCH FROM (ir.completed_at - ir.requested_at)
        )
    ) FILTER (
        WHERE ir.status = 'completed'
    ) AS average_execution_seconds
FROM neural_models nm
LEFT JOIN inference_requests ir
    ON ir.model_id = nm.model_id
GROUP BY nm.model_id, nm.model_name
ORDER BY nm.model_name;

-- ------------------------------------------------------------
-- 24. CTE: PARAMETER DISTRIBUTION
-- ------------------------------------------------------------

WITH parameter_counts AS (
    SELECT
        model_id,
        SUM(
            input_size * output_size + output_size
        ) AS total_parameters,
        COUNT(*) AS total_layers
    FROM dense_layers
    GROUP BY model_id
)
SELECT
    nm.model_name,
    pc.total_layers,
    pc.total_parameters,
    nm.input_features,
    nm.output_features
FROM neural_models nm
JOIN parameter_counts pc
    ON pc.model_id = nm.model_id
ORDER BY pc.total_parameters DESC;

-- ------------------------------------------------------------
-- 25. ELIGIBILITY DETERMINATION
-- ------------------------------------------------------------

SELECT
    nm.model_name,
    CASE
        WHEN nm.status <> 'active'
            THEN 'INELIGIBLE_MODEL_NOT_ACTIVE'
        WHEN EXISTS (
            SELECT 1
            FROM inference_requests ir
            WHERE ir.model_id = nm.model_id
              AND ir.status = 'pending'
        )
            THEN 'PENDING_INFERENCE_EXISTS'
        ELSE 'MODEL_AVAILABLE'
    END AS eligibility_status
FROM neural_models nm;

-- ------------------------------------------------------------
-- 26. FAILURE-CASE DEMONSTRATION
-- ------------------------------------------------------------

-- The following transaction demonstrates rejection of a model
-- with an input dimension mismatch without stopping the script.

DO $$
DECLARE
    v_model_id BIGINT;
BEGIN
    SELECT model_id
    INTO v_model_id
    FROM neural_network.neural_models
    WHERE model_name = 'invalid_dimension_example';

    BEGIN
        PERFORM neural_network.validate_model(v_model_id);

        RAISE EXCEPTION
            'Invalid model was unexpectedly accepted.';

    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE
            'Expected validation failure: %',
            SQLERRM;
    END;
END;
$$;

-- ------------------------------------------------------------
-- 27. SOFTMAX VERIFICATION
-- ------------------------------------------------------------

SELECT
    apply_activation(
        ARRAY[1.0, 2.0, 3.0]::DOUBLE PRECISION[],
        'softmax'
    ) AS normalized_probabilities;

-- ------------------------------------------------------------
-- 28. DATABASE INTEGRITY CHECK
-- ------------------------------------------------------------

SELECT
    nm.model_name,
    dl.layer_name,
    dl.input_size,
    dl.output_size,
    jsonb_array_length(lp.weights) AS weight_rows,
    cardinality(lp.biases) AS bias_count
FROM neural_models nm
JOIN dense_layers dl
    ON dl.model_id = nm.model_id
JOIN layer_parameters lp
    ON lp.layer_id = dl.layer_id
ORDER BY nm.model_name, dl.layer_index;

COMMIT;

-- ============================================================
-- END OF FORWARD PROPAGATION DATABASE IMPLEMENTATION
-- ============================================================
