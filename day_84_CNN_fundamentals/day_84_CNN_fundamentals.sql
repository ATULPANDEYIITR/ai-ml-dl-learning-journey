-- ============================================================
-- CNN FUNDAMENTALS
-- Convolution, Filters, Kernels, Stride and Padding
-- PostgreSQL 15+
-- ============================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS cnn;

SET search_path TO cnn, public;

-- ============================================================
-- 1. DOMAIN TYPES
-- ============================================================

CREATE TYPE padding_mode AS ENUM (
    'VALID',
    'SAME'
);

CREATE TYPE experiment_status AS ENUM (
    'CREATED',
    'RUNNING',
    'COMPLETED',
    'FAILED'
);

CREATE TYPE activation_type AS ENUM (
    'NONE',
    'RELU',
    'SIGMOID',
    'TANH'
);

-- ============================================================
-- 2. EXPERIMENT MANAGEMENT
-- ============================================================

CREATE TABLE experiments (
    experiment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    experiment_name VARCHAR(150) NOT NULL,

    description TEXT,

    input_channels INTEGER NOT NULL
        CHECK (input_channels > 0),

    input_height INTEGER NOT NULL
        CHECK (input_height > 0),

    input_width INTEGER NOT NULL
        CHECK (input_width > 0),

    status experiment_status NOT NULL DEFAULT 'CREATED',

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    completed_at TIMESTAMPTZ,

    CONSTRAINT valid_completion_time CHECK (
        (status = 'COMPLETED' AND completed_at IS NOT NULL)
        OR
        (status <> 'COMPLETED' AND completed_at IS NULL)
    )
);

-- ============================================================
-- 3. CONVOLUTION LAYER CONFIGURATION
-- ============================================================

CREATE TABLE convolution_layers (
    layer_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,

    layer_number INTEGER NOT NULL
        CHECK (layer_number >= 1),

    kernel_height INTEGER NOT NULL
        CHECK (kernel_height > 0),

    kernel_width INTEGER NOT NULL
        CHECK (kernel_width > 0),

    number_of_filters INTEGER NOT NULL
        CHECK (number_of_filters > 0),

    stride INTEGER NOT NULL DEFAULT 1
        CHECK (stride > 0),

    padding padding_mode NOT NULL DEFAULT 'VALID',

    activation activation_type NOT NULL DEFAULT 'RELU',

    input_channels INTEGER NOT NULL
        CHECK (input_channels > 0),

    input_height INTEGER NOT NULL
        CHECK (input_height > 0),

    input_width INTEGER NOT NULL
        CHECK (input_width > 0),

    output_height INTEGER NOT NULL
        CHECK (output_height > 0),

    output_width INTEGER NOT NULL
        CHECK (output_width > 0),

    trainable_parameters BIGINT GENERATED ALWAYS AS (
        number_of_filters::BIGINT *
        (
            input_channels::BIGINT *
            kernel_height *
            kernel_width + 1
        )
    ) STORED,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (experiment_id, layer_number)
);

CREATE INDEX idx_layers_experiment
    ON convolution_layers(experiment_id, layer_number);

-- ============================================================
-- 4. FILTER DEFINITIONS
-- ============================================================

CREATE TABLE filters (
    filter_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    layer_id BIGINT NOT NULL
        REFERENCES convolution_layers(layer_id)
        ON DELETE CASCADE,

    filter_number INTEGER NOT NULL
        CHECK (filter_number >= 1),

    bias DOUBLE PRECISION NOT NULL DEFAULT 0,

    description VARCHAR(200),

    UNIQUE (layer_id, filter_number),

    CHECK (
        bias NOT IN (
            'Infinity'::DOUBLE PRECISION,
            '-Infinity'::DOUBLE PRECISION,
            'NaN'::DOUBLE PRECISION
        )
    )
);

-- ============================================================
-- 5. KERNEL WEIGHTS
-- ============================================================

CREATE TABLE kernel_weights (
    weight_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    filter_id BIGINT NOT NULL
        REFERENCES filters(filter_id)
        ON DELETE CASCADE,

    input_channel INTEGER NOT NULL
        CHECK (input_channel >= 0),

    kernel_row INTEGER NOT NULL
        CHECK (kernel_row >= 0),

    kernel_column INTEGER NOT NULL
        CHECK (kernel_column >= 0),

    weight_value DOUBLE PRECISION NOT NULL,

    UNIQUE (
        filter_id,
        input_channel,
        kernel_row,
        kernel_column
    ),

    CHECK (
        weight_value NOT IN (
            'Infinity'::DOUBLE PRECISION,
            '-Infinity'::DOUBLE PRECISION,
            'NaN'::DOUBLE PRECISION
        )
    )
);

CREATE INDEX idx_kernel_filter
    ON kernel_weights(filter_id, input_channel);

-- ============================================================
-- 6. INPUT TENSOR VALUES
-- ============================================================

CREATE TABLE tensor_values (
    tensor_value_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,

    input_channel INTEGER NOT NULL
        CHECK (input_channel >= 0),

    row_index INTEGER NOT NULL
        CHECK (row_index >= 0),

    column_index INTEGER NOT NULL
        CHECK (column_index >= 0),

    value DOUBLE PRECISION NOT NULL,

    UNIQUE (
        experiment_id,
        input_channel,
        row_index,
        column_index
    ),

    CHECK (
        value NOT IN (
            'Infinity'::DOUBLE PRECISION,
            '-Infinity'::DOUBLE PRECISION,
            'NaN'::DOUBLE PRECISION
        )
    )
);

CREATE INDEX idx_tensor_coordinates
    ON tensor_values(
        experiment_id,
        input_channel,
        row_index,
        column_index
    );

-- ============================================================
-- 7. CONVOLUTION RESULTS
-- ============================================================

CREATE TABLE feature_map_values (
    feature_value_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    layer_id BIGINT NOT NULL
        REFERENCES convolution_layers(layer_id)
        ON DELETE CASCADE,

    filter_number INTEGER NOT NULL,

    output_row INTEGER NOT NULL
        CHECK (output_row >= 0),

    output_column INTEGER NOT NULL
        CHECK (output_column >= 0),

    raw_value DOUBLE PRECISION NOT NULL,

    activated_value DOUBLE PRECISION NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (
        layer_id,
        filter_number,
        output_row,
        output_column
    )
);

CREATE INDEX idx_feature_map_layer
    ON feature_map_values(
        layer_id,
        filter_number,
        output_row,
        output_column
    );

-- ============================================================
-- 8. OUTPUT SHAPE CALCULATION
-- ============================================================

CREATE OR REPLACE FUNCTION calculate_output_dimension(
    p_input INTEGER,
    p_kernel INTEGER,
    p_stride INTEGER,
    p_padding padding_mode
)
RETURNS INTEGER
LANGUAGE plpgsql
IMMUTABLE
STRICT
AS $$
BEGIN
    IF p_input <= 0 OR p_kernel <= 0 OR p_stride <= 0 THEN
        RAISE EXCEPTION
            'Input, kernel and stride must be positive.';
    END IF;

    IF p_padding = 'SAME' THEN
        RETURN (p_input + p_stride - 1) / p_stride;
    END IF;

    IF p_kernel > p_input THEN
        RAISE EXCEPTION
            'Kernel cannot exceed input under VALID padding.';
    END IF;

    RETURN ((p_input - p_kernel) / p_stride) + 1;
END;
$$;


CREATE OR REPLACE FUNCTION calculate_padding(
    p_input INTEGER,
    p_kernel INTEGER,
    p_stride INTEGER,
    p_padding padding_mode
)
RETURNS TABLE (
    padding_before INTEGER,
    padding_after INTEGER
)
LANGUAGE plpgsql
IMMUTABLE
STRICT
AS $$
DECLARE
    v_output INTEGER;
    v_total INTEGER;
BEGIN
    IF p_input <= 0 OR p_kernel <= 0 OR p_stride <= 0 THEN
        RAISE EXCEPTION 'Invalid padding dimensions.';
    END IF;

    IF p_padding = 'VALID' THEN
        IF p_kernel > p_input THEN
            RAISE EXCEPTION 'Kernel exceeds input.';
        END IF;

        RETURN QUERY SELECT 0, 0;
        RETURN;
    END IF;

    v_output := (p_input + p_stride - 1) / p_stride;

    v_total := GREATEST(
        0,
        (v_output - 1) * p_stride + p_kernel - p_input
    );

    padding_before := v_total / 2;
    padding_after := v_total - padding_before;

    RETURN NEXT;
END;
$$;

-- ============================================================
-- 9. LAYER CONFIGURATION VALIDATION
-- ============================================================

CREATE OR REPLACE FUNCTION validate_layer_configuration()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_expected_height INTEGER;
    v_expected_width INTEGER;
BEGIN
    SELECT input_channels, input_height, input_width
    INTO
        STRICT
        v_expected_height,
        v_expected_width,
        v_expected_height
    FROM experiments
    WHERE experiment_id = NEW.experiment_id;

    -- The experiment defines the first layer's input shape.
    IF NEW.layer_number = 1 THEN
        SELECT
            input_channels,
            input_height,
            input_width
        INTO
            v_expected_height,
            v_expected_width,
            v_expected_height
        FROM experiments
        WHERE experiment_id = NEW.experiment_id;
    END IF;

    IF NEW.output_height <> calculate_output_dimension(
        NEW.input_height,
        NEW.kernel_height,
        NEW.stride,
        NEW.padding
    ) THEN
        RAISE EXCEPTION 'Incorrect output height.';
    END IF;

    IF NEW.output_width <> calculate_output_dimension(
        NEW.input_width,
        NEW.kernel_width,
        NEW.stride,
        NEW.padding
    ) THEN
        RAISE EXCEPTION 'Incorrect output width.';
    END IF;

    IF NEW.layer_number = 1 THEN
        IF NOT EXISTS (
            SELECT 1
            FROM experiments e
            WHERE e.experiment_id = NEW.experiment_id
              AND e.input_channels = NEW.input_channels
              AND e.input_height = NEW.input_height
              AND e.input_width = NEW.input_width
        ) THEN
            RAISE EXCEPTION
                'First layer dimensions must match experiment input.';
        END IF;
    ELSE
        IF NOT EXISTS (
            SELECT 1
            FROM convolution_layers previous
            WHERE previous.experiment_id = NEW.experiment_id
              AND previous.layer_number = NEW.layer_number - 1
              AND previous.number_of_filters = NEW.input_channels
              AND previous.output_height = NEW.input_height
              AND previous.output_width = NEW.input_width
        ) THEN
            RAISE EXCEPTION
                'Layer input must match preceding layer output.';
        END IF;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_layer
BEFORE INSERT OR UPDATE ON convolution_layers
FOR EACH ROW
EXECUTE FUNCTION validate_layer_configuration();

-- ============================================================
-- 10. FILTER WEIGHT VALIDATION
-- ============================================================

CREATE OR REPLACE FUNCTION validate_kernel_weight()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_layer_id BIGINT;
    v_channels INTEGER;
    v_height INTEGER;
    v_width INTEGER;
BEGIN
    SELECT
        f.layer_id,
        l.input_channels,
        l.kernel_height,
        l.kernel_width
    INTO
        v_layer_id,
        v_channels,
        v_height,
        v_width
    FROM filters f
    JOIN convolution_layers l
      ON l.layer_id = f.layer_id
    WHERE f.filter_id = NEW.filter_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Filter does not exist.';
    END IF;

    IF NEW.input_channel >= v_channels THEN
        RAISE EXCEPTION 'Invalid input channel index.';
    END IF;

    IF NEW.kernel_row >= v_height THEN
        RAISE EXCEPTION 'Invalid kernel row.';
    END IF;

    IF NEW.kernel_column >= v_width THEN
        RAISE EXCEPTION 'Invalid kernel column.';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_kernel_weight
BEFORE INSERT OR UPDATE ON kernel_weights
FOR EACH ROW
EXECUTE FUNCTION validate_kernel_weight();

-- ============================================================
-- 11. INPUT TENSOR COORDINATE VALIDATION
-- ============================================================

CREATE OR REPLACE FUNCTION validate_tensor_coordinate()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_channels INTEGER;
    v_height INTEGER;
    v_width INTEGER;
BEGIN
    SELECT
        input_channels,
        input_height,
        input_width
    INTO
        v_channels,
        v_height,
        v_width
    FROM experiments
    WHERE experiment_id = NEW.experiment_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Experiment does not exist.';
    END IF;

    IF NEW.input_channel >= v_channels
       OR NEW.row_index >= v_height
       OR NEW.column_index >= v_width THEN

        RAISE EXCEPTION 'Tensor coordinate is outside input dimensions.';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_tensor_coordinate
BEFORE INSERT OR UPDATE ON tensor_values
FOR EACH ROW
EXECUTE FUNCTION validate_tensor_coordinate();

-- ============================================================
-- 12. EXPERIMENT STATE TRANSITIONS
-- ============================================================

CREATE OR REPLACE FUNCTION validate_experiment_transition()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.status = 'COMPLETED' AND NEW.status <> 'COMPLETED' THEN
        RAISE EXCEPTION 'Completed experiments are immutable in status.';
    END IF;

    IF OLD.status = 'FAILED'
       AND NEW.status NOT IN ('FAILED', 'CREATED') THEN
        RAISE EXCEPTION 'Invalid transition from FAILED.';
    END IF;

    IF OLD.status = 'RUNNING'
       AND NEW.status NOT IN ('RUNNING', 'COMPLETED', 'FAILED') THEN
        RAISE EXCEPTION 'Invalid transition from RUNNING.';
    END IF;

    IF NEW.status = 'COMPLETED' THEN
        NEW.completed_at := CURRENT_TIMESTAMP;
    ELSE
        NEW.completed_at := NULL;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_experiment_transition
BEFORE UPDATE OF status ON experiments
FOR EACH ROW
EXECUTE FUNCTION validate_experiment_transition();

-- ============================================================
-- 13. ELIGIBILITY VALIDATION
-- ============================================================

CREATE OR REPLACE FUNCTION check_experiment_eligibility(
    p_experiment_id BIGINT
)
RETURNS TABLE (
    eligible BOOLEAN,
    reason TEXT,
    configured_layers BIGINT,
    configured_filters BIGINT,
    supplied_input_values BIGINT
)
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_experiment experiments%ROWTYPE;
    v_layers BIGINT;
    v_filters BIGINT;
    v_inputs BIGINT;
BEGIN
    SELECT *
    INTO v_experiment
    FROM experiments
    WHERE experiment_id = p_experiment_id;

    IF NOT FOUND THEN
        RETURN QUERY SELECT
            FALSE,
            'Experiment not found'::TEXT,
            0::BIGINT,
            0::BIGINT,
            0::BIGINT;

        RETURN;
    END IF;

    SELECT COUNT(*)
    INTO v_layers
    FROM convolution_layers
    WHERE experiment_id = p_experiment_id;

    SELECT COUNT(*)
    INTO v_filters
    FROM filters f
    JOIN convolution_layers l
      ON l.layer_id = f.layer_id
    WHERE l.experiment_id = p_experiment_id;

    SELECT COUNT(*)
    INTO v_inputs
    FROM tensor_values
    WHERE experiment_id = p_experiment_id;

    IF v_layers = 0 THEN
        RETURN QUERY SELECT
            FALSE,
            'No convolution layer configured'::TEXT,
            v_layers,
            v_filters,
            v_inputs;

        RETURN;
    END IF;

    IF v_filters = 0 THEN
        RETURN QUERY SELECT
            FALSE,
            'No filters configured'::TEXT,
            v_layers,
            v_filters,
            v_inputs;

        RETURN;
    END IF;

    IF v_inputs <>
        (
            v_experiment.input_channels::BIGINT *
            v_experiment.input_height *
            v_experiment.input_width
        ) THEN

        RETURN QUERY SELECT
            FALSE,
            'Input tensor is incomplete'::TEXT,
            v_layers,
            v_filters,
            v_inputs;

        RETURN;
    END IF;

    RETURN QUERY SELECT
        TRUE,
        'Experiment is configured for execution'::TEXT,
        v_layers,
        v_filters,
        v_inputs;
END;
$$;

-- ============================================================
-- 14. CONVOLUTION RESULT AGGREGATION
-- ============================================================

CREATE OR REPLACE VIEW feature_map_statistics AS
SELECT
    l.experiment_id,
    l.layer_id,
    f.filter_number,
    COUNT(v.feature_value_id) AS output_elements,
    MIN(v.raw_value) AS minimum_raw_value,
    MAX(v.raw_value) AS maximum_raw_value,
    AVG(v.raw_value) AS mean_raw_value,
    AVG(v.activated_value) AS mean_activated_value,
    COUNT(*) FILTER (
        WHERE v.raw_value < 0
    ) AS negative_elements,
    COUNT(*) FILTER (
        WHERE v.activated_value = 0
    ) AS zero_after_activation
FROM convolution_layers l
JOIN filters f
  ON f.layer_id = l.layer_id
LEFT JOIN feature_map_values v
  ON v.layer_id = l.layer_id
 AND v.filter_number = f.filter_number
GROUP BY
    l.experiment_id,
    l.layer_id,
    f.filter_number;

-- ============================================================
-- 15. EXPERIMENT REPORTING
-- ============================================================

CREATE OR REPLACE VIEW experiment_report AS
SELECT
    e.experiment_id,
    e.experiment_name,
    e.status,
    e.input_channels,
    e.input_height,
    e.input_width,
    COUNT(DISTINCT l.layer_id) AS layer_count,
    COUNT(DISTINCT f.filter_id) AS filter_count,
    COALESCE(SUM(l.trainable_parameters), 0) AS total_parameters,
    COALESCE(SUM(v.output_elements), 0) AS stored_output_elements,
    e.created_at,
    e.completed_at
FROM experiments e
LEFT JOIN convolution_layers l
  ON l.experiment_id = e.experiment_id
LEFT JOIN filters f
  ON f.layer_id = l.layer_id
LEFT JOIN feature_map_statistics v
  ON v.layer_id = l.layer_id
 AND v.filter_number = f.filter_number
GROUP BY e.experiment_id;

-- ============================================================
-- 16. SAMPLE EXPERIMENT
-- ============================================================

INSERT INTO experiments (
    experiment_name,
    description,
    input_channels,
    input_height,
    input_width
)
VALUES (
    'Edge Detection Demonstration',
    'A single-channel convolution experiment using an edge-detection filter.',
    1,
    5,
    5
);

INSERT INTO convolution_layers (
    experiment_id,
    layer_number,
    kernel_height,
    kernel_width,
    number_of_filters,
    stride,
    padding,
    activation,
    input_channels,
    input_height,
    input_width,
    output_height,
    output_width
)
SELECT
    experiment_id,
    1,
    3,
    3,
    1,
    1,
    'VALID',
    'RELU',
    1,
    5,
    5,
    3,
    3
FROM experiments
WHERE experiment_name = 'Edge Detection Demonstration';

INSERT INTO filters (
    layer_id,
    filter_number,
    bias,
    description
)
SELECT
    layer_id,
    1,
    0,
    'Vertical edge detector'
FROM convolution_layers
WHERE layer_number = 1;

INSERT INTO kernel_weights (
    filter_id,
    input_channel,
    kernel_row,
    kernel_column,
    weight_value
)
SELECT
    f.filter_id,
    0,
    coordinates.row_index,
    coordinates.column_index,
    coordinates.weight_value
FROM filters f
CROSS JOIN (
    VALUES
        (0, 0, -1.0),
        (0, 1,  0.0),
        (0, 2,  1.0),
        (1, 0, -1.0),
        (1, 1,  0.0),
        (1, 2,  1.0),
        (2, 0, -1.0),
        (2, 1,  0.0),
        (2, 2,  1.0)
) AS coordinates(row_index, column_index, weight_value)
WHERE f.description = 'Vertical edge detector';

INSERT INTO tensor_values (
    experiment_id,
    input_channel,
    row_index,
    column_index,
    value
)
SELECT
    e.experiment_id,
    0,
    coordinates.row_index,
    coordinates.column_index,
    coordinates.value
FROM experiments e
CROSS JOIN (
    VALUES
        (0, 0, 1.0), (0, 1, 1.0), (0, 2, 1.0), (0, 3, 0.0), (0, 4, 0.0),
        (1, 0, 1.0), (1, 1, 1.0), (1, 2, 1.0), (1, 3, 0.0), (1, 4, 0.0),
        (2, 0, 1.0), (2, 1, 1.0), (2, 2, 1.0), (2, 3, 0.0), (2, 4, 0.0),
        (3, 0, 0.0), (3, 1, 0.0), (3, 2, 0.0), (3, 3, 1.0), (3, 4, 1.0),
        (4, 0, 0.0), (4, 1, 0.0), (4, 2, 0.0), (4, 3, 1.0), (4, 4, 1.0)
) AS coordinates(row_index, column_index, value)
WHERE e.experiment_name = 'Edge Detection Demonstration';

-- ============================================================
-- 17. CALCULATE AND STORE CONVOLUTION OUTPUT
-- ============================================================

CREATE OR REPLACE PROCEDURE execute_convolution(
    p_experiment_id BIGINT
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_layer convolution_layers%ROWTYPE;
    v_filter filters%ROWTYPE;
    v_total DOUBLE PRECISION;
    v_input_value DOUBLE PRECISION;
    v_weight DOUBLE PRECISION;
    v_row INTEGER;
    v_col INTEGER;
    v_channel INTEGER;
    v_kr INTEGER;
    v_kc INTEGER;
    v_pad_top INTEGER;
    v_pad_left INTEGER;
    v_expected_weights BIGINT;
    v_actual_weights BIGINT;
    v_eligible BOOLEAN;
BEGIN
    SELECT eligible
    INTO v_eligible
    FROM check_experiment_eligibility(p_experiment_id);

    IF NOT COALESCE(v_eligible, FALSE) THEN
        RAISE EXCEPTION
            'Experiment % is not eligible for execution.',
            p_experiment_id;
    END IF;

    UPDATE experiments
    SET status = 'RUNNING'
    WHERE experiment_id = p_experiment_id
      AND status = 'CREATED';

    IF NOT FOUND THEN
        RAISE EXCEPTION
            'Experiment must be in CREATED state.';
    END IF;

    FOR v_layer IN
        SELECT *
        FROM convolution_layers
        WHERE experiment_id = p_experiment_id
        ORDER BY layer_number
    LOOP

        SELECT padding_before
        INTO v_pad_top
        FROM calculate_padding(
            v_layer.input_height,
            v_layer.kernel_height,
            v_layer.stride,
            v_layer.padding
        );

        SELECT padding_before
        INTO v_pad_left
        FROM calculate_padding(
            v_layer.input_width,
            v_layer.kernel_width,
            v_layer.stride,
            v_layer.padding
        );

        FOR v_filter IN
            SELECT *
            FROM filters
            WHERE layer_id = v_layer.layer_id
            ORDER BY filter_number
        LOOP

            v_expected_weights :=
                v_layer.input_channels::BIGINT *
                v_layer.kernel_height *
                v_layer.kernel_width;

            SELECT COUNT(*)
            INTO v_actual_weights
            FROM kernel_weights
            WHERE filter_id = v_filter.filter_id;

            IF v_actual_weights <> v_expected_weights THEN
                RAISE EXCEPTION
                    'Filter % has incomplete kernel weights.',
                    v_filter.filter_number;
            END IF;

            FOR v_row IN 0..v_layer.output_height - 1 LOOP
                FOR v_col IN 0..v_layer.output_width - 1 LOOP

                    v_total := v_filter.bias;

                    FOR v_channel IN 0..v_layer.input_channels - 1 LOOP
                        FOR v_kr IN 0..v_layer.kernel_height - 1 LOOP
                            FOR v_kc IN 0..v_layer.kernel_width - 1 LOOP

                                SELECT COALESCE(MAX(tv.value), 0)
                                INTO v_input_value
                                FROM tensor_values tv
                                WHERE tv.experiment_id = p_experiment_id
                                  AND tv.input_channel = v_channel
                                  AND tv.row_index =
                                      v_row * v_layer.stride +
                                      v_kr - v_pad_top
                                  AND tv.column_index =
                                      v_col * v_layer.stride +
                                      v_kc - v_pad_left;

                                SELECT kw.weight_value
                                INTO STRICT v_weight
                                FROM kernel_weights kw
                                WHERE kw.filter_id = v_filter.filter_id
                                  AND kw.input_channel = v_channel
                                  AND kw.kernel_row = v_kr
                                  AND kw.kernel_column = v_kc;

                                v_total := v_total +
                                    v_input_value * v_weight;

                            END LOOP;
                        END LOOP;
                    END LOOP;

                    IF v_layer.activation = 'RELU' THEN
                        INSERT INTO feature_map_values (
                            layer_id,
                            filter_number,
                            output_row,
                            output_column,
                            raw_value,
                            activated_value
                        )
                        VALUES (
                            v_layer.layer_id,
                            v_filter.filter_number,
                            v_row,
                            v_col,
                            v_total,
                            GREATEST(0, v_total)
                        );

                    ELSIF v_layer.activation = 'SIGMOID' THEN
                        INSERT INTO feature_map_values (
                            layer_id,
                            filter_number,
                            output_row,
                            output_column,
                            raw_value,
                            activated_value
                        )
                        VALUES (
                            v_layer.layer_id,
                            v_filter.filter_number,
                            v_row,
                            v_col,
                            v_total,
                            CASE
                                WHEN v_total >= 0 THEN
                                    1.0 / (1.0 + EXP(-v_total))
                                ELSE
                                    EXP(v_total) / (1.0 + EXP(v_total))
                            END
                        );

                    ELSE
                        INSERT INTO feature_map_values (
                            layer_id,
                            filter_number,
                            output_row,
                            output_column,
                            raw_value,
                            activated_value
                        )
                        VALUES (
                            v_layer.layer_id,
                            v_filter.filter_number,
                            v_row,
                            v_col,
                            v_total,
                            v_total
                        );
                    END IF;

                END LOOP;
            END LOOP;
        END LOOP;
    END LOOP;

    UPDATE experiments
    SET status = 'COMPLETED'
    WHERE experiment_id = p_experiment_id;

EXCEPTION
    WHEN OTHERS THEN
        UPDATE experiments
        SET status = 'FAILED'
        WHERE experiment_id = p_experiment_id
          AND status = 'RUNNING';

        RAISE;
END;
$$;

-- ============================================================
-- 18. EXECUTE SAMPLE EXPERIMENT
-- ============================================================

CALL execute_convolution(
    (
        SELECT experiment_id
        FROM experiments
        WHERE experiment_name = 'Edge Detection Demonstration'
    )
);

-- ============================================================
-- 19. RETRIEVAL QUERIES
-- ============================================================

-- Experiment configuration.
SELECT *
FROM experiments
ORDER BY experiment_id;

-- Convolution layer configuration.
SELECT
    experiment_id,
    layer_number,
    input_channels,
    input_height,
    input_width,
    kernel_height,
    kernel_width,
    number_of_filters,
    stride,
    padding,
    output_height,
    output_width,
    trainable_parameters
FROM convolution_layers
ORDER BY experiment_id, layer_number;

-- Filter weights.
SELECT
    f.filter_number,
    kw.input_channel,
    kw.kernel_row,
    kw.kernel_column,
    kw.weight_value
FROM filters f
JOIN kernel_weights kw
  ON kw.filter_id = f.filter_id
ORDER BY
    f.filter_number,
    kw.input_channel,
    kw.kernel_row,
    kw.kernel_column;

-- Feature maps.
SELECT
    filter_number,
    output_row,
    output_column,
    raw_value,
    activated_value
FROM feature_map_values
ORDER BY filter_number, output_row, output_column;

-- ============================================================
-- 20. VALIDATION QUERIES
-- ============================================================

SELECT *
FROM check_experiment_eligibility(
    (
        SELECT experiment_id
        FROM experiments
        WHERE experiment_name = 'Edge Detection Demonstration'
    )
);

-- Verify expected feature map element counts.
SELECT
    l.layer_id,
    f.filter_number,
    COUNT(v.feature_value_id) AS actual_elements,
    l.output_height * l.output_width AS expected_elements,
    COUNT(v.feature_value_id) =
        l.output_height * l.output_width AS valid
FROM convolution_layers l
JOIN filters f
  ON f.layer_id = l.layer_id
LEFT JOIN feature_map_values v
  ON v.layer_id = l.layer_id
 AND v.filter_number = f.filter_number
GROUP BY
    l.layer_id,
    f.filter_number,
    l.output_height,
    l.output_width;

-- ============================================================
-- 21. AGGREGATION AND REPORTING
-- ============================================================

SELECT *
FROM feature_map_statistics
ORDER BY experiment_id, layer_id, filter_number;

SELECT *
FROM experiment_report
ORDER BY experiment_id;

-- ============================================================
-- 22. OUTPUT SHAPE EXAMPLES
-- ============================================================

SELECT
    calculate_output_dimension(32, 3, 1, 'VALID') AS valid_stride_1,
    calculate_output_dimension(32, 3, 1, 'SAME') AS same_stride_1,
    calculate_output_dimension(32, 3, 2, 'VALID') AS valid_stride_2,
    calculate_output_dimension(32, 3, 2, 'SAME') AS same_stride_2;

-- ============================================================
-- 23. FAILURE CASE DEMONSTRATION
-- ============================================================

-- Invalid output shape is rejected by the layer validation trigger.
-- The nested exception block rolls back only this demonstration.

DO $$
DECLARE
    v_experiment_id BIGINT;
BEGIN
    SELECT experiment_id
    INTO v_experiment_id
    FROM experiments
    WHERE experiment_name = 'Edge Detection Demonstration';

    BEGIN
        INSERT INTO convolution_layers (
            experiment_id,
            layer_number,
            kernel_height,
            kernel_width,
            number_of_filters,
            stride,
            padding,
            activation,
            input_channels,
            input_height,
            input_width,
            output_height,
            output_width
        )
        VALUES (
            v_experiment_id,
            2,
            3,
            3,
            4,
            1,
            'VALID',
            'RELU',
            1,
            5,
            5,
            99,
            99
        );

        RAISE EXCEPTION
            'Invalid output shape was unexpectedly accepted.';

    EXCEPTION
        WHEN OTHERS THEN
            RAISE NOTICE
                'Expected validation failure: %',
                SQLERRM;
    END;
END;
$$;

-- ============================================================
-- 24. TRANSACTIONAL INTEGRITY
-- ============================================================

-- All sample records, execution results and reports are committed
-- together. A failure in the main transaction prevents partial
-- persistence of the demonstration.

COMMIT;
