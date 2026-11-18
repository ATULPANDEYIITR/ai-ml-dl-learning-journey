-- ============================================================
-- PYTORCH FUNDAMENTALS
-- Tensor storage, devices, operations, autograd and experiments
--
-- PostgreSQL 15+
--
-- This database models tensor computation experiments.
-- Numerical tensor computation remains the responsibility of
-- the application execution engine.
--
-- Run:
--   psql -U postgres -d your_database -f pytorch_fundamentals.sql
-- ============================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS torch_lab;

SET search_path TO torch_lab, public;

-- ============================================================
-- 1. ENUMERATIONS
-- ============================================================

CREATE TYPE tensor_device AS ENUM (
    'cpu',
    'cuda'
);

CREATE TYPE tensor_dtype AS ENUM (
    'float16',
    'bfloat16',
    'float32',
    'float64',
    'int32',
    'int64'
);

CREATE TYPE tensor_status AS ENUM (
    'created',
    'validated',
    'executing',
    'completed',
    'failed',
    'cancelled'
);

CREATE TYPE operation_type AS ENUM (
    'creation',
    'addition',
    'subtraction',
    'multiplication',
    'division',
    'matrix_multiplication',
    'transpose',
    'reshape',
    'sum',
    'mean',
    'relu',
    'softmax',
    'backward'
);

CREATE TYPE gradient_status AS ENUM (
    'not_required',
    'pending',
    'computed',
    'failed'
);

CREATE TYPE experiment_status AS ENUM (
    'created',
    'running',
    'completed',
    'failed',
    'cancelled'
);

-- ============================================================
-- 2. USERS AND COMPUTE DEVICES
-- ============================================================

CREATE TABLE researchers (
    researcher_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    username VARCHAR(100) NOT NULL UNIQUE,

    email VARCHAR(255) NOT NULL UNIQUE,

    role VARCHAR(30) NOT NULL
        CHECK (
            role IN (
                'administrator',
                'researcher',
                'operator',
                'viewer'
            )
        ),

    active BOOLEAN NOT NULL DEFAULT TRUE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE compute_devices (
    device_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    device_name VARCHAR(150) NOT NULL UNIQUE,

    device_type tensor_device NOT NULL,

    hardware_model VARCHAR(150),

    total_memory_bytes BIGINT NOT NULL
        CHECK (total_memory_bytes > 0),

    available_memory_bytes BIGINT NOT NULL
        CHECK (
            available_memory_bytes >= 0
            AND available_memory_bytes <= total_memory_bytes
        ),

    driver_version VARCHAR(100),

    active BOOLEAN NOT NULL DEFAULT TRUE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- 3. EXPERIMENTS
-- ============================================================

CREATE TABLE experiments (
    experiment_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    experiment_name VARCHAR(200) NOT NULL,

    description TEXT,

    researcher_id BIGINT NOT NULL
        REFERENCES researchers(researcher_id),

    device_id BIGINT
        REFERENCES compute_devices(device_id),

    status experiment_status NOT NULL DEFAULT 'created',

    random_seed INTEGER NOT NULL DEFAULT 42,

    started_at TIMESTAMPTZ,

    completed_at TIMESTAMPTZ,

    failure_message TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_experiment_dates CHECK (
        completed_at IS NULL
        OR (
            started_at IS NOT NULL
            AND completed_at >= started_at
        )
    ),

    CONSTRAINT valid_failure_state CHECK (
        status <> 'failed'
        OR failure_message IS NOT NULL
    )
);

CREATE INDEX idx_experiments_researcher
    ON experiments(researcher_id, created_at DESC);

CREATE INDEX idx_experiments_status
    ON experiments(status, created_at DESC);

-- ============================================================
-- 4. TENSORS
-- ============================================================

CREATE TABLE tensors (
    tensor_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,

    tensor_name VARCHAR(150) NOT NULL,

    shape INTEGER[] NOT NULL,

    dtype tensor_dtype NOT NULL,

    device_id BIGINT NOT NULL
        REFERENCES compute_devices(device_id),

    requires_grad BOOLEAN NOT NULL DEFAULT FALSE,

    is_leaf BOOLEAN NOT NULL DEFAULT TRUE,

    gradient_state gradient_status NOT NULL
        DEFAULT 'not_required',

    element_count BIGINT NOT NULL
        CHECK (element_count > 0),

    estimated_bytes BIGINT NOT NULL
        CHECK (estimated_bytes > 0),

    finite_values BOOLEAN NOT NULL DEFAULT TRUE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_tensor_shape CHECK (
        cardinality(shape) > 0
        AND array_position(shape, NULL) IS NULL
        AND 0 < ALL(shape)
    ),

    CONSTRAINT valid_gradient_state CHECK (
        (
            requires_grad = FALSE
            AND gradient_state = 'not_required'
        )
        OR
        (
            requires_grad = TRUE
            AND gradient_state <> 'not_required'
        )
    ),

    CONSTRAINT unique_tensor_name
        UNIQUE(experiment_id, tensor_name)
);

CREATE INDEX idx_tensors_experiment
    ON tensors(experiment_id);

CREATE INDEX idx_tensors_device
    ON tensors(device_id, dtype);

CREATE INDEX idx_tensors_gradient
    ON tensors(gradient_state)
    WHERE requires_grad = TRUE;

-- ============================================================
-- 5. TENSOR VALUE VALIDATION
-- ============================================================

CREATE FUNCTION validate_tensor_metadata()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    calculated_elements NUMERIC := 1;
    dimension_value INTEGER;
    device_active BOOLEAN;
    device_memory BIGINT;
    dtype_size INTEGER;
BEGIN
    FOREACH dimension_value IN ARRAY NEW.shape
    LOOP
        IF dimension_value IS NULL OR dimension_value <= 0 THEN
            RAISE EXCEPTION
                'Invalid tensor dimension: %',
                dimension_value;
        END IF;

        calculated_elements :=
            calculated_elements * dimension_value;

        IF calculated_elements > 1000000000000 THEN
            RAISE EXCEPTION
                'Tensor exceeds configured element limit.';
        END IF;
    END LOOP;

    IF calculated_elements <> NEW.element_count THEN
        RAISE EXCEPTION
            'Shape describes % elements but metadata specifies %.',
            calculated_elements,
            NEW.element_count;
    END IF;

    SELECT active, total_memory_bytes
    INTO device_active, device_memory
    FROM compute_devices
    WHERE device_id = NEW.device_id;

    IF NOT FOUND OR NOT device_active THEN
        RAISE EXCEPTION
            'Selected compute device is unavailable.';
    END IF;

    dtype_size := CASE NEW.dtype
        WHEN 'float16' THEN 2
        WHEN 'bfloat16' THEN 2
        WHEN 'float32' THEN 4
        WHEN 'float64' THEN 8
        WHEN 'int32' THEN 4
        WHEN 'int64' THEN 8
    END;

    IF NEW.estimated_bytes <
        NEW.element_count * dtype_size THEN
        RAISE EXCEPTION
            'Estimated tensor storage is smaller than required storage.';
    END IF;

    IF NEW.estimated_bytes > device_memory THEN
        RAISE EXCEPTION
            'Tensor exceeds device memory capacity.';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_tensor_metadata
BEFORE INSERT OR UPDATE OF
    shape,
    dtype,
    device_id,
    element_count,
    estimated_bytes
ON tensors
FOR EACH ROW
EXECUTE FUNCTION validate_tensor_metadata();

-- ============================================================
-- 6. COMPUTATION GRAPH OPERATIONS
-- ============================================================

CREATE TABLE tensor_operations (
    operation_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,

    operation_name operation_type NOT NULL,

    input_tensor_id BIGINT NOT NULL
        REFERENCES tensors(tensor_id),

    output_tensor_id BIGINT NOT NULL
        REFERENCES tensors(tensor_id),

    operation_sequence INTEGER NOT NULL
        CHECK (operation_sequence > 0),

    execution_duration_ms NUMERIC(14, 4)
        CHECK (
            execution_duration_ms IS NULL
            OR execution_duration_ms >= 0
        ),

    operation_metadata JSONB NOT NULL DEFAULT '{}'::JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(experiment_id, operation_sequence),

    CHECK (input_tensor_id <> output_tensor_id)
);

CREATE INDEX idx_operations_experiment_sequence
    ON tensor_operations(
        experiment_id,
        operation_sequence
    );

CREATE INDEX idx_operations_type
    ON tensor_operations(operation_name);

-- Prevent accidental cross-experiment graph connections.
CREATE FUNCTION validate_operation_relationship()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    input_experiment BIGINT;
    output_experiment BIGINT;
BEGIN
    SELECT experiment_id
    INTO input_experiment
    FROM tensors
    WHERE tensor_id = NEW.input_tensor_id;

    SELECT experiment_id
    INTO output_experiment
    FROM tensors
    WHERE tensor_id = NEW.output_tensor_id;

    IF input_experiment <> NEW.experiment_id
       OR output_experiment <> NEW.experiment_id THEN
        RAISE EXCEPTION
            'Operation tensors must belong to the same experiment.';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_operation_relationship
BEFORE INSERT OR UPDATE
ON tensor_operations
FOR EACH ROW
EXECUTE FUNCTION validate_operation_relationship();

-- ============================================================
-- 7. GRADIENT RECORDS
-- ============================================================

CREATE TABLE tensor_gradients (
    gradient_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    tensor_id BIGINT NOT NULL
        REFERENCES tensors(tensor_id)
        ON DELETE CASCADE,

    backward_operation_id BIGINT NOT NULL
        REFERENCES tensor_operations(operation_id),

    gradient_norm DOUBLE PRECISION
        CHECK (
            gradient_norm IS NULL
            OR (
                gradient_norm >= 0
                AND gradient_norm < 'Infinity'::DOUBLE PRECISION
            )
        ),

    finite_gradient BOOLEAN NOT NULL,

    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(tensor_id, backward_operation_id)
);

CREATE INDEX idx_tensor_gradients_tensor
    ON tensor_gradients(tensor_id, computed_at DESC);

-- ============================================================
-- 8. PERFORMANCE AND VALIDATION METRICS
-- ============================================================

CREATE TABLE execution_metrics (
    metric_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,

    metric_name VARCHAR(100) NOT NULL,

    metric_value DOUBLE PRECISION NOT NULL,

    metric_unit VARCHAR(40),

    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CHECK (
        metric_value > '-Infinity'::DOUBLE PRECISION
        AND metric_value < 'Infinity'::DOUBLE PRECISION
    )
);

CREATE INDEX idx_execution_metrics_experiment
    ON execution_metrics(
        experiment_id,
        metric_name,
        recorded_at DESC
    );

-- ============================================================
-- 9. EXPERIMENT STATE TRANSITIONS
-- ============================================================

CREATE FUNCTION enforce_experiment_state_transition()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.status = NEW.status THEN
        RETURN NEW;
    END IF;

    IF NOT (
        (OLD.status = 'created'
            AND NEW.status IN ('running', 'cancelled'))
        OR
        (OLD.status = 'running'
            AND NEW.status IN (
                'completed',
                'failed',
                'cancelled'
            ))
    ) THEN
        RAISE EXCEPTION
            'Invalid experiment transition: % -> %',
            OLD.status,
            NEW.status;
    END IF;

    IF NEW.status = 'running' THEN
        NEW.started_at := COALESCE(NEW.started_at, NOW());
        NEW.completed_at := NULL;
        NEW.failure_message := NULL;
    END IF;

    IF NEW.status IN ('completed', 'failed', 'cancelled') THEN
        NEW.completed_at := NOW();
    END IF;

    IF NEW.status = 'failed'
       AND NEW.failure_message IS NULL THEN
        RAISE EXCEPTION
            'A failed experiment requires a failure message.';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_experiment_state_transition
BEFORE UPDATE OF status
ON experiments
FOR EACH ROW
EXECUTE FUNCTION enforce_experiment_state_transition();

-- ============================================================
-- 10. EXPERIMENT EXECUTION PROCEDURE
-- ============================================================

CREATE PROCEDURE start_experiment(
    p_experiment_id BIGINT
)
LANGUAGE plpgsql
AS $$
DECLARE
    current_status experiment_status;
BEGIN
    SELECT status
    INTO current_status
    FROM experiments
    WHERE experiment_id = p_experiment_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION
            'Experiment % does not exist.',
            p_experiment_id;
    END IF;

    IF current_status <> 'created' THEN
        RAISE EXCEPTION
            'Only created experiments can start.';
    END IF;

    UPDATE experiments
    SET status = 'running'
    WHERE experiment_id = p_experiment_id;
END;
$$;

-- ============================================================
-- 11. COMPUTE DEVICE MEMORY REPORT
-- ============================================================

CREATE VIEW device_memory_report AS
SELECT
    d.device_id,
    d.device_name,
    d.device_type,
    d.hardware_model,
    d.total_memory_bytes,
    d.available_memory_bytes,

    COALESCE(
        SUM(t.estimated_bytes),
        0
    ) AS registered_tensor_bytes,

    COUNT(t.tensor_id) AS registered_tensor_count,

    ROUND(
        100.0 * d.available_memory_bytes /
        NULLIF(d.total_memory_bytes, 0),
        2
    ) AS available_memory_percentage

FROM compute_devices d

LEFT JOIN tensors t
    ON t.device_id = d.device_id

GROUP BY
    d.device_id,
    d.device_name,
    d.device_type,
    d.hardware_model,
    d.total_memory_bytes,
    d.available_memory_bytes;

-- ============================================================
-- 12. TENSOR GRADIENT READINESS
-- ============================================================

CREATE VIEW gradient_readiness AS
SELECT
    t.tensor_id,
    t.tensor_name,
    t.requires_grad,
    t.gradient_state,

    COUNT(g.gradient_id) AS recorded_backward_passes,

    BOOL_AND(
        COALESCE(g.finite_gradient, TRUE)
    ) AS gradients_are_finite

FROM tensors t

LEFT JOIN tensor_gradients g
    ON g.tensor_id = t.tensor_id

GROUP BY
    t.tensor_id,
    t.tensor_name,
    t.requires_grad,
    t.gradient_state;

-- ============================================================
-- 13. SEED DATA
-- ============================================================

INSERT INTO researchers (
    username,
    email,
    role
)
VALUES
    (
        'research_admin',
        'admin@example.org',
        'administrator'
    ),
    (
        'tensor_researcher',
        'researcher@example.org',
        'researcher'
    ),
    (
        'experiment_operator',
        'operator@example.org',
        'operator'
    ),
    (
        'read_only_user',
        'viewer@example.org',
        'viewer'
    );

INSERT INTO compute_devices (
    device_name,
    device_type,
    hardware_model,
    total_memory_bytes,
    available_memory_bytes,
    driver_version
)
VALUES
    (
        'CPU-RESEARCH-01',
        'cpu',
        'AMD EPYC',
        68719476736,
        34359738368,
        NULL
    ),
    (
        'GPU-RESEARCH-01',
        'cuda',
        'NVIDIA GPU',
        25769803776,
        12884901888,
        '12.x'
    );

INSERT INTO experiments (
    experiment_name,
    description,
    researcher_id,
    device_id,
    random_seed
)
SELECT
    'tensor-autograd-baseline',
    'Tensor creation and gradient propagation experiment',
    r.researcher_id,
    d.device_id,
    42
FROM researchers r
CROSS JOIN compute_devices d
WHERE r.username = 'tensor_researcher'
  AND d.device_name = 'CPU-RESEARCH-01';

INSERT INTO experiments (
    experiment_name,
    description,
    researcher_id,
    device_id,
    random_seed
)
SELECT
    'cuda-device-evaluation',
    'GPU tensor execution and device allocation evaluation',
    r.researcher_id,
    d.device_id,
    123
FROM researchers r
CROSS JOIN compute_devices d
WHERE r.username = 'tensor_researcher'
  AND d.device_name = 'GPU-RESEARCH-01';

-- Failed experiment representing a non-finite computation.
INSERT INTO experiments (
    experiment_name,
    description,
    researcher_id,
    device_id,
    status,
    started_at,
    completed_at,
    failure_message
)
SELECT
    'unstable-gradient-test',
    'Experiment terminated after a non-finite loss',
    r.researcher_id,
    d.device_id,
    'failed',
    NOW() - INTERVAL '10 minutes',
    NOW(),
    'Non-finite loss detected.'
FROM researchers r
CROSS JOIN compute_devices d
WHERE r.username = 'tensor_researcher'
  AND d.device_name = 'CPU-RESEARCH-01';

-- ============================================================
-- 14. TENSOR METADATA
-- ============================================================

INSERT INTO tensors (
    experiment_id,
    tensor_name,
    shape,
    dtype,
    device_id,
    requires_grad,
    is_leaf,
    gradient_state,
    element_count,
    estimated_bytes
)
SELECT
    e.experiment_id,
    x.tensor_name,
    x.shape,
    x.dtype::tensor_dtype,
    d.device_id,
    x.requires_grad,
    x.is_leaf,
    x.gradient_state::gradient_status,
    x.element_count,
    x.estimated_bytes
FROM experiments e
JOIN compute_devices d
    ON d.device_id = e.device_id
CROSS JOIN (
    VALUES
        (
            'input_features',
            ARRAY[32, 4],
            'float32',
            TRUE,
            TRUE,
            'pending',
            128::BIGINT,
            512::BIGINT
        ),
        (
            'weight_matrix',
            ARRAY[4, 8],
            'float32',
            TRUE,
            TRUE,
            'pending',
            32::BIGINT,
            128::BIGINT
        ),
        (
            'activation_output',
            ARRAY[32, 8],
            'float32',
            TRUE,
            FALSE,
            'pending',
            256::BIGINT,
            1024::BIGINT
        ),
        (
            'inference_output',
            ARRAY[32, 8],
            'float32',
            FALSE,
            FALSE,
            'not_required',
            256::BIGINT,
            1024::BIGINT
        )
) AS x(
    tensor_name,
    shape,
    dtype,
    requires_grad,
    is_leaf,
    gradient_state,
    element_count,
    estimated_bytes
)
WHERE e.experiment_name = 'tensor-autograd-baseline'
  AND d.device_name = 'CPU-RESEARCH-01';

INSERT INTO tensors (
    experiment_id,
    tensor_name,
    shape,
    dtype,
    device_id,
    requires_grad,
    is_leaf,
    gradient_state,
    element_count,
    estimated_bytes
)
SELECT
    e.experiment_id,
    'gpu_batch',
    ARRAY[1024, 128],
    'float32',
    d.device_id,
    FALSE,
    TRUE,
    'not_required',
    131072,
    524288
FROM experiments e
JOIN compute_devices d
    ON d.device_id = e.device_id
WHERE e.experiment_name = 'cuda-device-evaluation';

-- ============================================================
-- 15. COMPUTATION GRAPH
-- ============================================================

INSERT INTO tensor_operations (
    experiment_id,
    operation_name,
    input_tensor_id,
    output_tensor_id,
    operation_sequence,
    execution_duration_ms,
    operation_metadata
)
SELECT
    e.experiment_id,
    'matrix_multiplication',
    input_tensor.tensor_id,
    output_tensor.tensor_id,
    1,
    1.425,
    '{"description":"Input matrix multiplied by weights","dimensions":[32,4,8]}'::JSONB
FROM experiments e
JOIN tensors input_tensor
    ON input_tensor.experiment_id = e.experiment_id
   AND input_tensor.tensor_name = 'input_features'
JOIN tensors output_tensor
    ON output_tensor.experiment_id = e.experiment_id
   AND output_tensor.tensor_name = 'activation_output'
WHERE e.experiment_name = 'tensor-autograd-baseline';

INSERT INTO tensor_operations (
    experiment_id,
    operation_name,
    input_tensor_id,
    output_tensor_id,
    operation_sequence,
    execution_duration_ms,
    operation_metadata
)
SELECT
    e.experiment_id,
    'backward',
    input_tensor.tensor_id,
    output_tensor.tensor_id,
    2,
    0.875,
    '{"description":"Backward gradient propagation"}'::JSONB
FROM experiments e
JOIN tensors input_tensor
    ON input_tensor.experiment_id = e.experiment_id
   AND input_tensor.tensor_name = 'activation_output'
JOIN tensors output_tensor
    ON output_tensor.experiment_id = e.experiment_id
   AND output_tensor.tensor_name = 'input_features'
WHERE e.experiment_name = 'tensor-autograd-baseline';

-- ============================================================
-- 16. GRADIENT RECORDS
-- ============================================================

INSERT INTO tensor_gradients (
    tensor_id,
    backward_operation_id,
    gradient_norm,
    finite_gradient
)
SELECT
    t.tensor_id,
    o.operation_id,
    0.8235,
    TRUE
FROM tensors t
JOIN tensor_operations o
    ON o.experiment_id = t.experiment_id
WHERE t.tensor_name = 'input_features'
  AND o.operation_name = 'backward';

UPDATE tensors
SET gradient_state = 'computed'
WHERE tensor_name = 'input_features';

-- ============================================================
-- 17. EXECUTION METRICS
-- ============================================================

INSERT INTO execution_metrics (
    experiment_id,
    metric_name,
    metric_value,
    metric_unit
)
SELECT
    experiment_id,
    metric_name,
    metric_value,
    metric_unit
FROM experiments
CROSS JOIN (
    VALUES
        ('forward_time', 1.425, 'milliseconds'),
        ('backward_time', 0.875, 'milliseconds'),
        ('peak_memory', 2048.0, 'bytes'),
        ('gradient_norm', 0.8235, 'unitless')
) AS metrics(
    metric_name,
    metric_value,
    metric_unit
)
WHERE experiment_name = 'tensor-autograd-baseline';

-- ============================================================
-- 18. ELIGIBILITY FUNCTION
-- ============================================================

CREATE FUNCTION experiment_is_ready(
    p_experiment_id BIGINT
)
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    current_status experiment_status;
    tensor_count BIGINT;
    invalid_tensor_count BIGINT;
    device_available BOOLEAN;
BEGIN
    SELECT status
    INTO current_status
    FROM experiments
    WHERE experiment_id = p_experiment_id;

    IF NOT FOUND OR current_status <> 'running' THEN
        RETURN FALSE;
    END IF;

    SELECT COUNT(*)
    INTO tensor_count
    FROM tensors
    WHERE experiment_id = p_experiment_id;

    IF tensor_count = 0 THEN
        RETURN FALSE;
    END IF;

    SELECT COUNT(*)
    INTO invalid_tensor_count
    FROM tensors
    WHERE experiment_id = p_experiment_id
      AND (
          finite_values = FALSE
          OR element_count <= 0
          OR estimated_bytes <= 0
      );

    IF invalid_tensor_count > 0 THEN
        RETURN FALSE;
    END IF;

    SELECT d.active
    INTO device_available
    FROM experiments e
    JOIN compute_devices d
        ON d.device_id = e.device_id
    WHERE e.experiment_id = p_experiment_id;

    RETURN COALESCE(device_available, FALSE);
END;
$$;

-- ============================================================
-- 19. REPORTING QUERIES
-- ============================================================

-- A. Retrieve experiments and their compute devices.
SELECT
    e.experiment_id,
    e.experiment_name,
    e.status,
    r.username,
    d.device_name,
    d.device_type
FROM experiments e
JOIN researchers r
    ON r.researcher_id = e.researcher_id
LEFT JOIN compute_devices d
    ON d.device_id = e.device_id
ORDER BY e.created_at DESC;

-- B. Retrieve tensor metadata.
SELECT
    e.experiment_name,
    t.tensor_name,
    t.shape,
    t.dtype,
    t.element_count,
    t.estimated_bytes,
    t.requires_grad,
    t.gradient_state
FROM tensors t
JOIN experiments e
    ON e.experiment_id = t.experiment_id
ORDER BY e.experiment_id, t.tensor_id;

-- C. Computation graph sequence.
SELECT
    e.experiment_name,
    o.operation_sequence,
    o.operation_name,
    source.tensor_name AS input_tensor,
    destination.tensor_name AS output_tensor,
    o.execution_duration_ms
FROM tensor_operations o
JOIN experiments e
    ON e.experiment_id = o.experiment_id
JOIN tensors source
    ON source.tensor_id = o.input_tensor_id
JOIN tensors destination
    ON destination.tensor_id = o.output_tensor_id
ORDER BY e.experiment_id, o.operation_sequence;

-- D. Aggregate device utilization metadata.
SELECT *
FROM device_memory_report
ORDER BY device_name;

-- E. Gradient readiness.
SELECT *
FROM gradient_readiness
ORDER BY tensor_id;

-- F. Execution performance.
SELECT
    e.experiment_name,
    m.metric_name,
    AVG(m.metric_value) AS average_value,
    MAX(m.metric_value) AS maximum_value,
    COUNT(*) AS measurement_count
FROM execution_metrics m
JOIN experiments e
    ON e.experiment_id = m.experiment_id
GROUP BY e.experiment_name, m.metric_name
ORDER BY e.experiment_name, m.metric_name;

-- G. CTE-based experiment quality report.
WITH tensor_summary AS (
    SELECT
        experiment_id,
        COUNT(*) AS total_tensors,
        COUNT(*) FILTER (
            WHERE requires_grad
        ) AS differentiable_tensors,
        COUNT(*) FILTER (
            WHERE gradient_state = 'computed'
        ) AS computed_gradients,
        SUM(estimated_bytes) AS registered_memory
    FROM tensors
    GROUP BY experiment_id
),
operation_summary AS (
    SELECT
        experiment_id,
        COUNT(*) AS total_operations,
        SUM(execution_duration_ms) AS execution_time_ms
    FROM tensor_operations
    GROUP BY experiment_id
)
SELECT
    e.experiment_name,
    e.status,
    COALESCE(t.total_tensors, 0) AS total_tensors,
    COALESCE(t.differentiable_tensors, 0)
        AS differentiable_tensors,
    COALESCE(t.computed_gradients, 0)
        AS computed_gradients,
    COALESCE(t.registered_memory, 0)
        AS registered_memory,
    COALESCE(o.total_operations, 0)
        AS total_operations,
    COALESCE(o.execution_time_ms, 0)
        AS execution_time_ms
FROM experiments e
LEFT JOIN tensor_summary t
    ON t.experiment_id = e.experiment_id
LEFT JOIN operation_summary o
    ON o.experiment_id = e.experiment_id
ORDER BY e.experiment_id;

-- H. Check experiment execution eligibility.
SELECT
    experiment_id,
    experiment_name,
    status,
    experiment_is_ready(experiment_id) AS execution_eligible
FROM experiments;

-- ============================================================
-- 20. TRANSACTIONAL WORKFLOW EXAMPLE
-- ============================================================

SAVEPOINT before_execution;

-- Start the first created experiment.
SELECT start_experiment(experiment_id)
FROM experiments
WHERE experiment_name = 'tensor-autograd-baseline';

-- Verify execution readiness.
SELECT
    experiment_id,
    experiment_is_ready(experiment_id)
FROM experiments
WHERE experiment_name = 'tensor-autograd-baseline';

-- Complete the experiment.
UPDATE experiments
SET status = 'completed'
WHERE experiment_name = 'tensor-autograd-baseline'
  AND status = 'running';

RELEASE SAVEPOINT before_execution;

-- ============================================================
-- 21. FAILURE CASE VALIDATION
-- ============================================================

-- Invalid experiment transitions are rejected.
DO $$
DECLARE
    target_id BIGINT;
BEGIN
    SELECT experiment_id
    INTO target_id
    FROM experiments
    WHERE experiment_name = 'tensor-autograd-baseline';

    BEGIN
        UPDATE experiments
        SET status = 'running'
        WHERE experiment_id = target_id;

        RAISE EXCEPTION
            'Expected transition failure did not occur.';

    EXCEPTION
        WHEN OTHERS THEN
            IF SQLERRM =
                'Expected transition failure did not occur.' THEN
                RAISE;
            END IF;

            RAISE NOTICE
                'Expected state-transition rejection: %',
                SQLERRM;
    END;
END;
$$;

-- Invalid tensor shape is rejected.
DO $$
DECLARE
    target_experiment BIGINT;
    target_device BIGINT;
BEGIN
    SELECT e.experiment_id, e.device_id
    INTO target_experiment, target_device
    FROM experiments e
    WHERE e.experiment_name = 'tensor-autograd-baseline';

    BEGIN
        INSERT INTO tensors (
            experiment_id,
            tensor_name,
            shape,
            dtype,
            device_id,
            element_count,
            estimated_bytes
        )
        VALUES (
            target_experiment,
            'invalid_tensor',
            ARRAY[2, 3],
            'float32',
            target_device,
            5,
            20
        );

        RAISE EXCEPTION
            'Expected tensor validation failure did not occur.';

    EXCEPTION
        WHEN OTHERS THEN
            IF SQLERRM =
                'Expected tensor validation failure did not occur.' THEN
                RAISE;
            END IF;

            RAISE NOTICE
                'Expected tensor validation rejection: %',
                SQLERRM;
    END;
END;
$$;

COMMIT;

-- ============================================================
-- END OF PYTORCH FUNDAMENTALS DATABASE MODEL
-- ============================================================
