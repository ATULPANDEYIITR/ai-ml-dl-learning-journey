-- ============================================================
-- NEURAL NETWORK MATHEMATICS
-- PostgreSQL 14+
--
-- Domain:
-- Neurons, weighted sums, biases, activation functions,
-- network architecture, forward propagation, and training.
--
-- Includes:
-- Relational modeling, integrity constraints, mathematical
-- functions, stored procedures, triggers, sample data,
-- inference, and performance-oriented indexes.
-- ============================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS neural_math;

SET search_path TO neural_math, public;

-- ============================================================
-- 1. ENUMERATED DOMAIN TYPES
-- ============================================================

CREATE TYPE activation_type AS ENUM (
    'linear',
    'sigmoid',
    'tanh',
    'relu',
    'leaky_relu'
);

CREATE TYPE network_status AS ENUM (
    'draft',
    'configured',
    'training',
    'trained',
    'failed',
    'archived'
);

CREATE TYPE layer_type AS ENUM (
    'input',
    'hidden',
    'output'
);

CREATE TYPE execution_status AS ENUM (
    'pending',
    'running',
    'completed',
    'failed'
);

-- ============================================================
-- 2. NETWORK REGISTRY
-- ============================================================

CREATE TABLE neural_networks (
    network_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    network_name VARCHAR(150) NOT NULL UNIQUE,

    description TEXT NOT NULL DEFAULT '',

    status network_status NOT NULL DEFAULT 'draft',

    learning_rate DOUBLE PRECISION NOT NULL DEFAULT 0.01,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_learning_rate CHECK (
        learning_rate > 0
        AND learning_rate < 10
        AND learning_rate <> 'Infinity'::DOUBLE PRECISION
        AND learning_rate <> 'NaN'::DOUBLE PRECISION
    )
);

-- ============================================================
-- 3. LAYERS
-- ============================================================

CREATE TABLE network_layers (
    layer_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    network_id BIGINT NOT NULL
        REFERENCES neural_networks(network_id)
        ON DELETE CASCADE,

    layer_number INTEGER NOT NULL,

    layer_name VARCHAR(100) NOT NULL,

    layer_kind layer_type NOT NULL,

    neuron_count INTEGER NOT NULL,

    activation activation_type,

    CONSTRAINT unique_layer_position
        UNIQUE(network_id, layer_number),

    CONSTRAINT valid_layer_number
        CHECK (layer_number >= 0),

    CONSTRAINT valid_neuron_count
        CHECK (neuron_count > 0),

    CONSTRAINT input_layer_activation
        CHECK (
            (layer_kind = 'input' AND activation IS NULL)
            OR
            (layer_kind <> 'input' AND activation IS NOT NULL)
        )
);

CREATE INDEX idx_layers_network
    ON network_layers(network_id, layer_number);

-- ============================================================
-- 4. NEURONS
-- ============================================================

CREATE TABLE neurons (
    neuron_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    layer_id BIGINT NOT NULL
        REFERENCES network_layers(layer_id)
        ON DELETE CASCADE,

    neuron_number INTEGER NOT NULL,

    bias DOUBLE PRECISION NOT NULL DEFAULT 0,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_neuron_position
        UNIQUE(layer_id, neuron_number),

    CONSTRAINT valid_neuron_number
        CHECK (neuron_number >= 0),

    CONSTRAINT valid_bias CHECK (
        bias > '-Infinity'::DOUBLE PRECISION
        AND bias < 'Infinity'::DOUBLE PRECISION
        AND bias <> 'NaN'::DOUBLE PRECISION
    )
);

CREATE INDEX idx_neurons_layer
    ON neurons(layer_id, neuron_number);

-- ============================================================
-- 5. WEIGHT CONNECTIONS
-- ============================================================

CREATE TABLE neuron_weights (
    weight_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    source_neuron_id BIGINT NOT NULL
        REFERENCES neurons(neuron_id)
        ON DELETE CASCADE,

    target_neuron_id BIGINT NOT NULL
        REFERENCES neurons(neuron_id)
        ON DELETE CASCADE,

    weight_value DOUBLE PRECISION NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_neuron_connection
        UNIQUE(source_neuron_id, target_neuron_id),

    CONSTRAINT valid_weight CHECK (
        weight_value > '-Infinity'::DOUBLE PRECISION
        AND weight_value < 'Infinity'::DOUBLE PRECISION
        AND weight_value <> 'NaN'::DOUBLE PRECISION
    ),

    CONSTRAINT no_self_connection
        CHECK (source_neuron_id <> target_neuron_id)
);

CREATE INDEX idx_weights_target
    ON neuron_weights(target_neuron_id);

CREATE INDEX idx_weights_source
    ON neuron_weights(source_neuron_id);

-- ============================================================
-- 6. TRAINING RUNS
-- ============================================================

CREATE TABLE training_runs (
    training_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    network_id BIGINT NOT NULL
        REFERENCES neural_networks(network_id)
        ON DELETE CASCADE,

    status execution_status NOT NULL DEFAULT 'pending',

    epoch_count INTEGER NOT NULL DEFAULT 0,

    learning_rate DOUBLE PRECISION NOT NULL,

    initial_loss DOUBLE PRECISION,

    final_loss DOUBLE PRECISION,

    started_at TIMESTAMPTZ,

    completed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_epoch_count
        CHECK (epoch_count >= 0),

    CONSTRAINT valid_training_rate
        CHECK (
            learning_rate > 0
            AND learning_rate < 10
            AND learning_rate <> 'Infinity'::DOUBLE PRECISION
            AND learning_rate <> 'NaN'::DOUBLE PRECISION
        ),

    CONSTRAINT valid_initial_loss CHECK (
        initial_loss IS NULL OR (
            initial_loss >= 0
            AND initial_loss < 'Infinity'::DOUBLE PRECISION
            AND initial_loss <> 'NaN'::DOUBLE PRECISION
        )
    ),

    CONSTRAINT valid_final_loss CHECK (
        final_loss IS NULL OR (
            final_loss >= 0
            AND final_loss < 'Infinity'::DOUBLE PRECISION
            AND final_loss <> 'NaN'::DOUBLE PRECISION
        )
    )
);

CREATE INDEX idx_training_network_status
    ON training_runs(network_id, status);

-- ============================================================
-- 7. INFERENCE RECORDS
-- ============================================================

CREATE TABLE inference_records (
    inference_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    network_id BIGINT NOT NULL
        REFERENCES neural_networks(network_id)
        ON DELETE CASCADE,

    input_values DOUBLE PRECISION[] NOT NULL,

    output_values DOUBLE PRECISION[],

    execution_status execution_status NOT NULL DEFAULT 'pending',

    execution_time_ms NUMERIC(12, 4),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT nonempty_inputs
        CHECK (cardinality(input_values) > 0),

    CONSTRAINT nonempty_outputs
        CHECK (
            output_values IS NULL
            OR cardinality(output_values) > 0
        ),

    CONSTRAINT valid_execution_time
        CHECK (
            execution_time_ms IS NULL
            OR execution_time_ms >= 0
        )
);

CREATE INDEX idx_inference_network_date
    ON inference_records(network_id, created_at DESC);

-- ============================================================
-- 8. ACTIVATION FUNCTION
-- ============================================================

CREATE OR REPLACE FUNCTION activation_value(
    p_value DOUBLE PRECISION,
    p_activation activation_type
)
RETURNS DOUBLE PRECISION
LANGUAGE plpgsql
IMMUTABLE
STRICT
AS $$
DECLARE
    v_exp DOUBLE PRECISION;
BEGIN
    IF p_value = 'Infinity'::DOUBLE PRECISION
       OR p_value = '-Infinity'::DOUBLE PRECISION
       OR p_value = 'NaN'::DOUBLE PRECISION THEN

        RAISE EXCEPTION 'Activation input must be finite';
    END IF;

    CASE p_activation

        WHEN 'linear' THEN
            RETURN p_value;

        WHEN 'sigmoid' THEN
            IF p_value >= 0 THEN
                RETURN 1.0 / (
                    1.0 + EXP(-LEAST(p_value, 709.0))
                );
            ELSE
                v_exp := EXP(GREATEST(p_value, -709.0));

                RETURN v_exp / (1.0 + v_exp);
            END IF;

        WHEN 'tanh' THEN
            RETURN TANH(p_value);

        WHEN 'relu' THEN
            RETURN GREATEST(0.0, p_value);

        WHEN 'leaky_relu' THEN
            RETURN CASE
                WHEN p_value > 0 THEN p_value
                ELSE 0.01 * p_value
            END;

    END CASE;

    RAISE EXCEPTION 'Unsupported activation function';
END;
$$;

-- ============================================================
-- 9. ACTIVATION DERIVATIVE
-- ============================================================

CREATE OR REPLACE FUNCTION activation_derivative(
    p_value DOUBLE PRECISION,
    p_activation activation_type
)
RETURNS DOUBLE PRECISION
LANGUAGE plpgsql
IMMUTABLE
STRICT
AS $$
DECLARE
    v_output DOUBLE PRECISION;
BEGIN
    CASE p_activation

        WHEN 'linear' THEN
            RETURN 1.0;

        WHEN 'sigmoid' THEN
            v_output := activation_value(
                p_value,
                p_activation
            );

            RETURN v_output * (1.0 - v_output);

        WHEN 'tanh' THEN
            v_output := TANH(p_value);

            RETURN 1.0 - v_output * v_output;

        WHEN 'relu' THEN
            RETURN CASE WHEN p_value > 0 THEN 1.0 ELSE 0.0 END;

        WHEN 'leaky_relu' THEN
            RETURN CASE WHEN p_value > 0 THEN 1.0 ELSE 0.01 END;

    END CASE;

    RAISE EXCEPTION 'Unsupported activation derivative';
END;
$$;

-- ============================================================
-- 10. WEIGHTED SUM
-- ============================================================

CREATE OR REPLACE FUNCTION calculate_weighted_sum(
    p_inputs DOUBLE PRECISION[],
    p_weights DOUBLE PRECISION[],
    p_bias DOUBLE PRECISION
)
RETURNS DOUBLE PRECISION
LANGUAGE plpgsql
IMMUTABLE
STRICT
AS $$
DECLARE
    v_result DOUBLE PRECISION := p_bias;
    v_index INTEGER;
BEGIN
    IF cardinality(p_inputs) = 0 THEN
        RAISE EXCEPTION 'Inputs cannot be empty';
    END IF;

    IF cardinality(p_inputs) <> cardinality(p_weights) THEN
        RAISE EXCEPTION
            'Input and weight dimensions do not match';
    END IF;

    IF p_bias = 'Infinity'::DOUBLE PRECISION
       OR p_bias = '-Infinity'::DOUBLE PRECISION
       OR p_bias = 'NaN'::DOUBLE PRECISION THEN
        RAISE EXCEPTION 'Bias must be finite';
    END IF;

    FOR v_index IN 1..cardinality(p_inputs)
    LOOP
        IF p_inputs[v_index] = 'Infinity'::DOUBLE PRECISION
           OR p_inputs[v_index] = '-Infinity'::DOUBLE PRECISION
           OR p_inputs[v_index] = 'NaN'::DOUBLE PRECISION
           OR p_weights[v_index] = 'Infinity'::DOUBLE PRECISION
           OR p_weights[v_index] = '-Infinity'::DOUBLE PRECISION
           OR p_weights[v_index] = 'NaN'::DOUBLE PRECISION THEN

            RAISE EXCEPTION
                'Inputs and weights must be finite';
        END IF;

        v_result := v_result
            + p_inputs[v_index] * p_weights[v_index];
    END LOOP;

    IF v_result = 'Infinity'::DOUBLE PRECISION
       OR v_result = '-Infinity'::DOUBLE PRECISION
       OR v_result = 'NaN'::DOUBLE PRECISION THEN

        RAISE EXCEPTION 'Weighted sum overflow';
    END IF;

    RETURN v_result;
END;
$$;

-- ============================================================
-- 11. NETWORK STRUCTURE VALIDATION
-- ============================================================

CREATE OR REPLACE FUNCTION validate_network_structure(
    p_network_id BIGINT
)
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_layer_count INTEGER;
    v_input_count INTEGER;
    v_output_count INTEGER;
    v_min_layer INTEGER;
    v_max_layer INTEGER;
BEGIN
    SELECT
        COUNT(*),
        COUNT(*) FILTER (WHERE layer_kind = 'input'),
        COUNT(*) FILTER (WHERE layer_kind = 'output'),
        MIN(layer_number),
        MAX(layer_number)
    INTO
        v_layer_count,
        v_input_count,
        v_output_count,
        v_min_layer,
        v_max_layer
    FROM network_layers
    WHERE network_id = p_network_id;

    IF v_layer_count < 2
       OR v_input_count <> 1
       OR v_output_count <> 1 THEN
        RETURN FALSE;
    END IF;

    IF v_min_layer <> 0
       OR v_max_layer <> v_layer_count - 1 THEN
        RETURN FALSE;
    END IF;

    -- Layer numbers must be contiguous.
    IF EXISTS (
        SELECT 1
        FROM generate_series(0, v_layer_count - 1) AS expected(n)
        WHERE NOT EXISTS (
            SELECT 1
            FROM network_layers l
            WHERE l.network_id = p_network_id
              AND l.layer_number = expected.n
        )
    ) THEN
        RETURN FALSE;
    END IF;

    RETURN TRUE;
END;
$$;

-- ============================================================
-- 12. NEURON FORWARD PROPAGATION
-- ============================================================

CREATE OR REPLACE FUNCTION evaluate_neuron(
    p_neuron_id BIGINT,
    p_inputs DOUBLE PRECISION[]
)
RETURNS TABLE (
    weighted_sum DOUBLE PRECISION,
    activation_output DOUBLE PRECISION
)
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_bias DOUBLE PRECISION;
    v_activation activation_type;
    v_weights DOUBLE PRECISION[];
    v_sum DOUBLE PRECISION;
BEGIN
    SELECT n.bias, l.activation
    INTO v_bias, v_activation
    FROM neurons n
    JOIN network_layers l ON l.layer_id = n.layer_id
    WHERE n.neuron_id = p_neuron_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Neuron % does not exist', p_neuron_id;
    END IF;

    SELECT ARRAY_AGG(
        w.weight_value
        ORDER BY source_n.neuron_number
    )
    INTO v_weights
    FROM neuron_weights w
    JOIN neurons source_n
        ON source_n.neuron_id = w.source_neuron_id
    WHERE w.target_neuron_id = p_neuron_id;

    IF v_weights IS NULL THEN
        RAISE EXCEPTION 'Neuron has no incoming weights';
    END IF;

    v_sum := calculate_weighted_sum(
        p_inputs,
        v_weights,
        v_bias
    );

    RETURN QUERY
    SELECT
        v_sum,
        activation_value(v_sum, v_activation);
END;
$$;

-- ============================================================
-- 13. TRAINING RUN STATE TRANSITIONS
-- ============================================================

CREATE OR REPLACE FUNCTION enforce_training_transitions()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.status IN ('completed', 'failed')
       AND NEW.status <> OLD.status THEN

        RAISE EXCEPTION
            'Completed or failed training runs cannot be restarted';
    END IF;

    IF OLD.status = 'pending'
       AND NEW.status NOT IN ('pending', 'running', 'failed') THEN

        RAISE EXCEPTION
            'Invalid pending training transition';
    END IF;

    IF OLD.status = 'running'
       AND NEW.status NOT IN ('running', 'completed', 'failed') THEN

        RAISE EXCEPTION
            'Invalid running training transition';
    END IF;

    IF NEW.status = 'running'
       AND NEW.started_at IS NULL THEN

        NEW.started_at := NOW();
    END IF;

    IF NEW.status IN ('completed', 'failed') THEN
        NEW.completed_at := NOW();
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_training_transitions
BEFORE UPDATE ON training_runs
FOR EACH ROW
EXECUTE FUNCTION enforce_training_transitions();

-- ============================================================
-- 14. NETWORK UPDATE TIMESTAMP
-- ============================================================

CREATE OR REPLACE FUNCTION update_network_timestamp()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_network_updated_at
BEFORE UPDATE ON neural_networks
FOR EACH ROW
EXECUTE FUNCTION update_network_timestamp();

-- ============================================================
-- 15. TRAINING REPORT VIEW
-- ============================================================

CREATE OR REPLACE VIEW training_summary AS
SELECT
    n.network_id,
    n.network_name,
    n.status AS network_status,
    COUNT(t.training_id) AS total_training_runs,
    COUNT(*) FILTER (
        WHERE t.status = 'completed'
    ) AS completed_runs,
    COUNT(*) FILTER (
        WHERE t.status = 'failed'
    ) AS failed_runs,
    AVG(t.final_loss) FILTER (
        WHERE t.status = 'completed'
    ) AS average_final_loss,
    MIN(t.final_loss) FILTER (
        WHERE t.status = 'completed'
    ) AS best_final_loss
FROM neural_networks n
LEFT JOIN training_runs t
    ON t.network_id = n.network_id
GROUP BY
    n.network_id,
    n.network_name,
    n.status;

-- ============================================================
-- 16. MERGE-FREE SAMPLE NETWORK CONFIGURATION
-- ============================================================

INSERT INTO neural_networks (
    network_name,
    description,
    status,
    learning_rate
)
VALUES
(
    'xor_classification',
    'Two-input nonlinear XOR classification network',
    'configured',
    0.5
),
(
    'binary_risk_classifier',
    'Binary classification demonstration network',
    'configured',
    0.01
);

-- XOR architecture: 2 -> 2 -> 1.

INSERT INTO network_layers (
    network_id,
    layer_number,
    layer_name,
    layer_kind,
    neuron_count,
    activation
)
SELECT
    network_id,
    layer_number,
    layer_name,
    layer_kind,
    neuron_count,
    activation
FROM neural_networks n
CROSS JOIN (
    VALUES
        (0, 'Input Layer', 'input'::layer_type, 2, NULL::activation_type),
        (1, 'Hidden Layer', 'hidden'::layer_type, 2, 'tanh'::activation_type),
        (2, 'Output Layer', 'output'::layer_type, 1, 'sigmoid'::activation_type)
) AS layers(
    layer_number,
    layer_name,
    layer_kind,
    neuron_count,
    activation
)
WHERE n.network_name = 'xor_classification';

-- Risk classifier architecture: 3 -> 2 -> 1.

INSERT INTO network_layers (
    network_id,
    layer_number,
    layer_name,
    layer_kind,
    neuron_count,
    activation
)
SELECT
    network_id,
    layer_number,
    layer_name,
    layer_kind,
    neuron_count,
    activation
FROM neural_networks n
CROSS JOIN (
    VALUES
        (0, 'Input Layer', 'input'::layer_type, 3, NULL::activation_type),
        (1, 'Hidden Layer', 'hidden'::layer_type, 2, 'relu'::activation_type),
        (2, 'Output Layer', 'output'::layer_type, 1, 'sigmoid'::activation_type)
) AS layers(
    layer_number,
    layer_name,
    layer_kind,
    neuron_count,
    activation
)
WHERE n.network_name = 'binary_risk_classifier';

-- ============================================================
-- 17. CREATE NEURONS
-- ============================================================

INSERT INTO neurons (
    layer_id,
    neuron_number,
    bias
)
SELECT
    l.layer_id,
    neuron_number,
    CASE
        WHEN l.layer_kind = 'hidden' THEN 0.1
        WHEN l.layer_kind = 'output' THEN 0.2
        ELSE 0.0
    END
FROM network_layers l
CROSS JOIN LATERAL
    generate_series(0, l.neuron_count - 1) AS neuron_number;

-- ============================================================
-- 18. CREATE WEIGHT CONNECTIONS
-- ============================================================

-- Connect every neuron in each preceding layer to every neuron
-- in the following layer.

INSERT INTO neuron_weights (
    source_neuron_id,
    target_neuron_id,
    weight_value
)
SELECT
    source_n.neuron_id,
    target_n.neuron_id,
    CASE
        WHEN source_l.network_id = (
            SELECT network_id
            FROM neural_networks
            WHERE network_name = 'xor_classification'
        )
        THEN
            CASE
                WHEN source_n.neuron_number = 0
                     AND target_n.neuron_number = 0 THEN 0.5
                WHEN source_n.neuron_number = 1
                     AND target_n.neuron_number = 0 THEN -0.4
                WHEN source_n.neuron_number = 0
                     AND target_n.neuron_number = 1 THEN -0.3
                WHEN source_n.neuron_number = 1
                     AND target_n.neuron_number = 1 THEN 0.7
                ELSE 0.8
            END
        ELSE
            0.25 * (
                source_n.neuron_number + target_n.neuron_number + 1
            )
    END
FROM neurons source_n
JOIN network_layers source_l
    ON source_l.layer_id = source_n.layer_id
JOIN network_layers target_l
    ON target_l.network_id = source_l.network_id
   AND target_l.layer_number = source_l.layer_number + 1
JOIN neurons target_n
    ON target_n.layer_id = target_l.layer_id;

-- ============================================================
-- 19. NETWORK STRUCTURE VALIDATION QUERIES
-- ============================================================

SELECT
    network_name,
    validate_network_structure(network_id) AS structure_valid
FROM neural_networks
ORDER BY network_id;

-- Layer and neuron counts.
SELECT
    n.network_name,
    l.layer_number,
    l.layer_kind,
    l.activation,
    COUNT(neuron.neuron_id) AS actual_neurons
FROM neural_networks n
JOIN network_layers l
    ON l.network_id = n.network_id
LEFT JOIN neurons neuron
    ON neuron.layer_id = l.layer_id
GROUP BY
    n.network_name,
    l.layer_number,
    l.layer_kind,
    l.activation
ORDER BY
    n.network_name,
    l.layer_number;

-- ============================================================
-- 20. SAMPLE SINGLE NEURON CALCULATION
-- ============================================================

-- The first hidden neuron in the XOR network.
WITH selected_neuron AS (
    SELECT neuron_id
    FROM neurons n
    JOIN network_layers l
        ON l.layer_id = n.layer_id
    JOIN neural_networks nn
        ON nn.network_id = l.network_id
    WHERE nn.network_name = 'xor_classification'
      AND l.layer_number = 1
      AND n.neuron_number = 0
)
SELECT
    weighted_sum,
    activation_output
FROM selected_neuron
CROSS JOIN LATERAL evaluate_neuron(
    selected_neuron.neuron_id,
    ARRAY[1.0, 0.0]::DOUBLE PRECISION[]
);

-- ============================================================
-- 21. TRAINING RUN RECORDS
-- ============================================================

INSERT INTO training_runs (
    network_id,
    status,
    epoch_count,
    learning_rate,
    initial_loss,
    final_loss,
    started_at,
    completed_at
)
SELECT
    network_id,
    'completed',
    2500,
    learning_rate,
    0.25,
    0.0008,
    NOW() - INTERVAL '30 minutes',
    NOW()
FROM neural_networks
WHERE network_name = 'xor_classification';

INSERT INTO training_runs (
    network_id,
    status,
    epoch_count,
    learning_rate,
    initial_loss,
    final_loss
)
SELECT
    network_id,
    'failed',
    15,
    learning_rate,
    0.9,
    0.95
FROM neural_networks
WHERE network_name = 'binary_risk_classifier';

-- ============================================================
-- 22. TRAINING WORKFLOW PROCEDURE
-- ============================================================

CREATE OR REPLACE PROCEDURE start_training_run(
    p_network_id BIGINT,
    p_epochs INTEGER,
    p_learning_rate DOUBLE PRECISION
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_training_id BIGINT;
BEGIN
    IF p_epochs <= 0 THEN
        RAISE EXCEPTION 'Epoch count must be positive';
    END IF;

    IF p_learning_rate <= 0
       OR p_learning_rate >= 10
       OR p_learning_rate = 'Infinity'::DOUBLE PRECISION
       OR p_learning_rate = 'NaN'::DOUBLE PRECISION THEN

        RAISE EXCEPTION 'Invalid learning rate';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM neural_networks
        WHERE network_id = p_network_id
          AND status IN ('configured', 'trained')
    ) THEN
        RAISE EXCEPTION
            'Network does not exist or cannot be trained';
    END IF;

    IF NOT validate_network_structure(p_network_id) THEN
        RAISE EXCEPTION 'Network structure is invalid';
    END IF;

    INSERT INTO training_runs (
        network_id,
        status,
        epoch_count,
        learning_rate
    )
    VALUES (
        p_network_id,
        'pending',
        p_epochs,
        p_learning_rate
    )
    RETURNING training_id INTO v_training_id;

    UPDATE training_runs
    SET status = 'running'
    WHERE training_id = v_training_id;

    UPDATE neural_networks
    SET status = 'training'
    WHERE network_id = p_network_id;

    RAISE NOTICE
        'Training run % started for network %',
        v_training_id,
        p_network_id;
END;
$$;

-- ============================================================
-- 23. TRAINING ELIGIBILITY
-- ============================================================

CREATE OR REPLACE FUNCTION network_training_eligible(
    p_network_id BIGINT
)
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1
        FROM neural_networks n
        WHERE n.network_id = p_network_id
          AND n.status IN ('configured', 'trained')
          AND validate_network_structure(n.network_id)
          AND NOT EXISTS (
              SELECT 1
              FROM training_runs t
              WHERE t.network_id = n.network_id
                AND t.status = 'running'
          )
    );
END;
$$;

SELECT
    network_name,
    network_training_eligible(network_id) AS training_eligible
FROM neural_networks;

-- ============================================================
-- 24. INFERENCE SAMPLE RECORDS
-- ============================================================

INSERT INTO inference_records (
    network_id,
    input_values,
    output_values,
    execution_status,
    execution_time_ms
)
SELECT
    network_id,
    ARRAY[0.0, 1.0]::DOUBLE PRECISION[],
    ARRAY[0.85]::DOUBLE PRECISION[],
    'completed',
    1.25
FROM neural_networks
WHERE network_name = 'xor_classification';

INSERT INTO inference_records (
    network_id,
    input_values,
    execution_status
)
SELECT
    network_id,
    ARRAY[0.5, 0.8, 0.2]::DOUBLE PRECISION[],
    'failed'
FROM neural_networks
WHERE network_name = 'binary_risk_classifier';

-- ============================================================
-- 25. AGGREGATION AND REPORTING
-- ============================================================

SELECT *
FROM training_summary
ORDER BY network_name;

-- Inference statistics.
SELECT
    n.network_name,
    COUNT(i.inference_id) AS total_inferences,
    COUNT(*) FILTER (
        WHERE i.execution_status = 'completed'
    ) AS successful_inferences,
    COUNT(*) FILTER (
        WHERE i.execution_status = 'failed'
    ) AS failed_inferences,
    AVG(i.execution_time_ms) AS average_execution_ms
FROM neural_networks n
LEFT JOIN inference_records i
    ON i.network_id = n.network_id
GROUP BY n.network_id, n.network_name
ORDER BY n.network_name;

-- ============================================================
-- 26. WEIGHT DISTRIBUTION ANALYSIS
-- ============================================================

SELECT
    nn.network_name,
    source_l.layer_number AS source_layer,
    target_l.layer_number AS target_layer,
    COUNT(w.weight_id) AS connection_count,
    AVG(w.weight_value) AS average_weight,
    MIN(w.weight_value) AS minimum_weight,
    MAX(w.weight_value) AS maximum_weight,
    STDDEV_POP(w.weight_value) AS weight_stddev
FROM neuron_weights w
JOIN neurons source_n
    ON source_n.neuron_id = w.source_neuron_id
JOIN neurons target_n
    ON target_n.neuron_id = w.target_neuron_id
JOIN network_layers source_l
    ON source_l.layer_id = source_n.layer_id
JOIN network_layers target_l
    ON target_l.layer_id = target_n.layer_id
JOIN neural_networks nn
    ON nn.network_id = source_l.network_id
GROUP BY
    nn.network_name,
    source_l.layer_number,
    target_l.layer_number
ORDER BY
    nn.network_name,
    source_layer;

-- ============================================================
-- 27. ACTIVATION FUNCTION EVALUATION
-- ============================================================

SELECT
    activation,
    activation_value(1.5, activation) AS output,
    activation_derivative(1.5, activation) AS derivative
FROM unnest(
    ARRAY[
        'linear'::activation_type,
        'sigmoid'::activation_type,
        'tanh'::activation_type,
        'relu'::activation_type,
        'leaky_relu'::activation_type
    ]
) AS activation;

-- ============================================================
-- 28. TRANSACTION FAILURE DEMONSTRATION
-- ============================================================

-- This transaction demonstrates atomic rollback when a
-- mathematical configuration violates an integrity rule.

SAVEPOINT before_invalid_weight;

DO $$
BEGIN
    BEGIN
        INSERT INTO neuron_weights (
            source_neuron_id,
            target_neuron_id,
            weight_value
        )
        VALUES (
            (SELECT MIN(neuron_id) FROM neurons),
            (SELECT MAX(neuron_id) FROM neurons),
            'NaN'::DOUBLE PRECISION
        );

    EXCEPTION WHEN check_violation THEN
        RAISE NOTICE
            'Invalid weight rejected by database constraint';
    END;
END;
$$;

ROLLBACK TO SAVEPOINT before_invalid_weight;

-- ============================================================
-- 29. RETRIEVAL: COMPLETE NETWORK CONFIGURATION
-- ============================================================

SELECT
    nn.network_name,
    l.layer_number,
    l.layer_name,
    l.layer_kind,
    l.activation,
    n.neuron_number,
    n.bias,
    COUNT(w.weight_id) AS incoming_connections
FROM neural_networks nn
JOIN network_layers l
    ON l.network_id = nn.network_id
JOIN neurons n
    ON n.layer_id = l.layer_id
LEFT JOIN neuron_weights w
    ON w.target_neuron_id = n.neuron_id
GROUP BY
    nn.network_name,
    l.layer_number,
    l.layer_name,
    l.layer_kind,
    l.activation,
    n.neuron_number,
    n.bias
ORDER BY
    nn.network_name,
    l.layer_number,
    n.neuron_number;

-- ============================================================
-- 30. COMPLETE TRANSACTION
-- ============================================================

COMMIT;
