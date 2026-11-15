-- ============================================================
-- BACKPROPAGATION AND COMPUTATIONAL GRAPH GOVERNANCE
-- PostgreSQL 18 compatible
--
-- Models:
-- Computational graphs
-- Operations and dependencies
-- Tensor metadata
-- Forward values
-- Gradient propagation
-- Training experiments
-- Optimization steps
-- Gradient validation
-- Training lifecycle
-- ============================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS backprop;

SET search_path TO backprop, public;

-- ------------------------------------------------------------
-- 1. Domain types
-- ------------------------------------------------------------

CREATE TYPE node_operation AS ENUM (
    'INPUT',
    'PARAMETER',
    'ADD',
    'MULTIPLY',
    'RELU',
    'TANH',
    'SIGMOID',
    'EXP',
    'LOG',
    'POWER',
    'MEAN',
    'SQUARE'
);

CREATE TYPE run_state AS ENUM (
    'CREATED',
    'VALIDATING',
    'FORWARD',
    'BACKWARD',
    'UPDATING',
    'COMPLETED',
    'FAILED'
);

CREATE TYPE gradient_state AS ENUM (
    'NOT_COMPUTED',
    'COMPUTED',
    'VERIFIED',
    'REJECTED'
);

-- ------------------------------------------------------------
-- 2. Repository of computational experiments
-- ------------------------------------------------------------

CREATE TABLE experiments (
    experiment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_name VARCHAR(150) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    input_dimensions INTEGER NOT NULL CHECK (input_dimensions > 0),
    learning_rate NUMERIC(18, 10) NOT NULL
        CHECK (learning_rate > 0),
    gradient_clip NUMERIC(18, 10) NOT NULL
        CHECK (gradient_clip > 0),
    seed INTEGER NOT NULL DEFAULT 42,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE computational_graphs (
    graph_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id),
    graph_name VARCHAR(150) NOT NULL,
    graph_version INTEGER NOT NULL DEFAULT 1
        CHECK (graph_version > 0),
    is_acyclic BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (experiment_id, graph_name, graph_version)
);

-- ------------------------------------------------------------
-- 3. Computational graph nodes
-- ------------------------------------------------------------

CREATE TABLE graph_nodes (
    node_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    graph_id BIGINT NOT NULL
        REFERENCES computational_graphs(graph_id)
        ON DELETE CASCADE,
    node_name VARCHAR(150) NOT NULL,
    operation node_operation NOT NULL,
    forward_value NUMERIC(30, 15),
    requires_gradient BOOLEAN NOT NULL DEFAULT FALSE,
    gradient_value NUMERIC(30, 15),
    gradient_status gradient_state NOT NULL DEFAULT 'NOT_COMPUTED',
    operation_exponent NUMERIC(18, 8),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (graph_id, node_name),
    UNIQUE (graph_id, node_id),
    CHECK (
        operation <> 'POWER'
        OR operation_exponent IS NOT NULL
    )
);

-- ------------------------------------------------------------
-- 4. Directed graph dependencies
--
-- A dependency means that the child operation consumes
-- the parent node's output.
-- ------------------------------------------------------------

CREATE TABLE graph_edges (
    edge_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    graph_id BIGINT NOT NULL,
    parent_node_id BIGINT NOT NULL,
    child_node_id BIGINT NOT NULL,
    input_position INTEGER NOT NULL CHECK (input_position >= 0),

    FOREIGN KEY (graph_id, parent_node_id)
        REFERENCES graph_nodes(graph_id, node_id)
        ON DELETE CASCADE,

    FOREIGN KEY (graph_id, child_node_id)
        REFERENCES graph_nodes(graph_id, node_id)
        ON DELETE CASCADE,

    CHECK (parent_node_id <> child_node_id),

    UNIQUE (child_node_id, input_position),
    UNIQUE (graph_id, parent_node_id, child_node_id)
);

CREATE INDEX idx_graph_edges_parent
    ON graph_edges(graph_id, parent_node_id);

CREATE INDEX idx_graph_edges_child
    ON graph_edges(graph_id, child_node_id);

CREATE INDEX idx_graph_nodes_operation
    ON graph_nodes(graph_id, operation);

-- ------------------------------------------------------------
-- 5. Tensor metadata
-- ------------------------------------------------------------

CREATE TABLE tensor_metadata (
    tensor_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    node_id BIGINT NOT NULL UNIQUE
        REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
    data_type VARCHAR(30) NOT NULL
        CHECK (data_type IN ('FLOAT32', 'FLOAT64')),
    shape INTEGER[] NOT NULL,
    element_count BIGINT NOT NULL CHECK (element_count > 0),
    device VARCHAR(30) NOT NULL DEFAULT 'CPU',
    CHECK (cardinality(shape) > 0),
    CHECK (array_position(shape, 0) IS NULL),
    CHECK (
        NOT EXISTS (
            SELECT 1
            FROM unnest(shape) AS dimension
            WHERE dimension < 1
        )
    )
);

-- ------------------------------------------------------------
-- 6. Training runs and optimization steps
-- ------------------------------------------------------------

CREATE TABLE training_runs (
    run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id),
    graph_id BIGINT NOT NULL
        REFERENCES computational_graphs(graph_id),
    state run_state NOT NULL DEFAULT 'CREATED',
    total_epochs INTEGER NOT NULL CHECK (total_epochs > 0),
    current_epoch INTEGER NOT NULL DEFAULT 0,
    current_loss NUMERIC(30, 15),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    failure_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (
        current_epoch >= 0
        AND current_epoch <= total_epochs
    ),
    CHECK (
        (state = 'FAILED' AND failure_reason IS NOT NULL)
        OR state <> 'FAILED'
    )
);

CREATE INDEX idx_training_runs_experiment
    ON training_runs(experiment_id, created_at DESC);

CREATE TABLE optimization_steps (
    step_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL
        REFERENCES training_runs(run_id) ON DELETE CASCADE,
    epoch INTEGER NOT NULL CHECK (epoch > 0),
    learning_rate NUMERIC(18, 10) NOT NULL
        CHECK (learning_rate > 0),
    loss_before NUMERIC(30, 15),
    loss_after NUMERIC(30, 15),
    gradient_norm NUMERIC(30, 15),
    gradient_clipped BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (run_id, epoch),
    CHECK (gradient_norm IS NULL OR gradient_norm >= 0)
);

-- ------------------------------------------------------------
-- 7. Gradient audit
-- ------------------------------------------------------------

CREATE TABLE gradient_records (
    gradient_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    step_id BIGINT NOT NULL
        REFERENCES optimization_steps(step_id) ON DELETE CASCADE,
    node_id BIGINT NOT NULL
        REFERENCES graph_nodes(node_id),
    analytical_gradient NUMERIC(30, 15) NOT NULL,
    numerical_gradient NUMERIC(30, 15),
    absolute_error NUMERIC(30, 15),
    verification_passed BOOLEAN,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (step_id, node_id),
    CHECK (absolute_error IS NULL OR absolute_error >= 0)
);

-- ------------------------------------------------------------
-- 8. Training dataset and samples
-- ------------------------------------------------------------

CREATE TABLE training_datasets (
    dataset_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id),
    dataset_name VARCHAR(150) NOT NULL,
    sample_count INTEGER NOT NULL CHECK (sample_count > 0),
    feature_count INTEGER NOT NULL CHECK (feature_count > 0),
    dataset_hash VARCHAR(128) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (experiment_id, dataset_name)
);

CREATE TABLE dataset_samples (
    sample_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL
        REFERENCES training_datasets(dataset_id) ON DELETE CASCADE,
    sample_number INTEGER NOT NULL CHECK (sample_number >= 0),
    features NUMERIC[] NOT NULL,
    target NUMERIC NOT NULL,
    UNIQUE (dataset_id, sample_number),
    CHECK (cardinality(features) > 0),
    CHECK (
        array_position(features, NULL) IS NULL
    )
);

CREATE INDEX idx_dataset_samples_dataset
    ON dataset_samples(dataset_id, sample_number);

-- ------------------------------------------------------------
-- 9. Graph integrity and cycle detection
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION validate_graph_edge()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    parent_graph BIGINT;
    child_graph BIGINT;
    cycle_found BOOLEAN;
BEGIN
    SELECT graph_id
    INTO parent_graph
    FROM graph_nodes
    WHERE node_id = NEW.parent_node_id;

    SELECT graph_id
    INTO child_graph
    FROM graph_nodes
    WHERE node_id = NEW.child_node_id;

    IF parent_graph IS NULL OR child_graph IS NULL THEN
        RAISE EXCEPTION 'Graph edge references a missing node';
    END IF;

    IF parent_graph <> child_graph
       OR parent_graph <> NEW.graph_id THEN
        RAISE EXCEPTION 'Cross-graph dependencies are prohibited';
    END IF;

    -- Follow existing outgoing dependencies from the proposed child.
    -- If the proposed parent is reachable, this edge creates a cycle.
    WITH RECURSIVE descendants(node_id) AS (
        SELECT NEW.child_node_id

        UNION

        SELECT edge.child_node_id
        FROM graph_edges edge
        JOIN descendants d
            ON edge.parent_node_id = d.node_id
        WHERE edge.graph_id = NEW.graph_id
    )
    SELECT EXISTS (
        SELECT 1
        FROM descendants
        WHERE node_id = NEW.parent_node_id
    )
    INTO cycle_found;

    IF cycle_found THEN
        RAISE EXCEPTION 'Computational graph cycle detected';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_graph_edge
BEFORE INSERT OR UPDATE ON graph_edges
FOR EACH ROW
EXECUTE FUNCTION validate_graph_edge();

-- ------------------------------------------------------------
-- 10. Operation arity validation
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION validate_operation_arity()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    edge_count INTEGER;
    expected_count INTEGER;
BEGIN
    SELECT COUNT(*)
    INTO edge_count
    FROM graph_edges
    WHERE child_node_id = NEW.node_id;

    expected_count := CASE
        WHEN NEW.operation IN ('INPUT', 'PARAMETER') THEN 0
        WHEN NEW.operation IN (
            'ADD', 'MULTIPLY'
        ) THEN 2
        ELSE 1
    END;

    IF edge_count <> expected_count THEN
        RAISE EXCEPTION
            'Operation % requires % inputs; found %',
            NEW.operation,
            expected_count,
            edge_count;
    END IF;

    RETURN NEW;
END;
$$;

-- ------------------------------------------------------------
-- 11. Graph evaluation function
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION calculate_node(
    p_operation node_operation,
    p_values NUMERIC[],
    p_exponent NUMERIC DEFAULT NULL
)
RETURNS NUMERIC
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    result NUMERIC;
    x NUMERIC;
    y NUMERIC;
BEGIN
    IF p_operation = 'INPUT' OR p_operation = 'PARAMETER' THEN
        IF cardinality(p_values) <> 1 THEN
            RAISE EXCEPTION 'Input node requires one value';
        END IF;

        RETURN p_values[1];
    END IF;

    IF p_operation IN ('ADD', 'MULTIPLY') THEN
        IF cardinality(p_values) <> 2 THEN
            RAISE EXCEPTION 'Binary operation requires two values';
        END IF;

        x := p_values[1];
        y := p_values[2];

        IF p_operation = 'ADD' THEN
            RETURN x + y;
        END IF;

        RETURN x * y;
    END IF;

    IF cardinality(p_values) <> 1 THEN
        RAISE EXCEPTION 'Unary operation requires one value';
    END IF;

    x := p_values[1];

    CASE p_operation
        WHEN 'SQUARE' THEN
            result := x * x;

        WHEN 'POWER' THEN
            IF p_exponent IS NULL THEN
                RAISE EXCEPTION 'POWER requires an exponent';
            END IF;

            IF x < 0 AND p_exponent <> trunc(p_exponent) THEN
                RAISE EXCEPTION 'Invalid fractional power';
            END IF;

            result := power(x, p_exponent);

        WHEN 'RELU' THEN
            result := greatest(0, x);

        WHEN 'TANH' THEN
            result := tanh(x);

        WHEN 'SIGMOID' THEN
            -- Stable sigmoid avoids overflow for large magnitudes.
            IF x >= 0 THEN
                result := 1 / (1 + exp(-x));
            ELSE
                result := exp(x) / (1 + exp(x));
            END IF;

        WHEN 'EXP' THEN
            result := exp(x);

        WHEN 'LOG' THEN
            IF x <= 0 THEN
                RAISE EXCEPTION 'LOG requires a positive value';
            END IF;

            result := ln(x);

        WHEN 'MEAN' THEN
            result := x;

        ELSE
            RAISE EXCEPTION 'Unsupported operation %', p_operation;
    END CASE;

    RETURN result;
END;
$$;

-- ------------------------------------------------------------
-- 12. Run-state validation
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION validate_training_transition()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.state IN ('COMPLETED', 'FAILED') THEN
        RAISE EXCEPTION 'Terminal training runs cannot be modified';
    END IF;

    IF NOT (
        (OLD.state = 'CREATED' AND NEW.state = 'VALIDATING')
        OR
        (OLD.state = 'VALIDATING' AND NEW.state = 'FORWARD')
        OR
        (OLD.state = 'FORWARD' AND NEW.state = 'BACKWARD')
        OR
        (OLD.state = 'BACKWARD' AND NEW.state = 'UPDATING')
        OR
        (OLD.state = 'UPDATING' AND NEW.state IN (
            'FORWARD', 'COMPLETED'
        ))
        OR
        (NEW.state = 'FAILED')
    ) THEN
        RAISE EXCEPTION
            'Invalid training transition: % -> %',
            OLD.state,
            NEW.state;
    END IF;

    IF NEW.state = 'COMPLETED'
       AND NEW.current_epoch <> NEW.total_epochs THEN
        RAISE EXCEPTION
            'Training cannot complete before all epochs finish';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_training_transition
BEFORE UPDATE OF state ON training_runs
FOR EACH ROW
EXECUTE FUNCTION validate_training_transition();

-- ------------------------------------------------------------
-- 13. Experiment evaluation view
-- ------------------------------------------------------------

CREATE VIEW experiment_training_report AS
SELECT
    e.experiment_id,
    e.experiment_name,
    COUNT(DISTINCT r.run_id) AS total_runs,
    COUNT(DISTINCT r.run_id)
        FILTER (WHERE r.state = 'COMPLETED') AS completed_runs,
    COUNT(DISTINCT r.run_id)
        FILTER (WHERE r.state = 'FAILED') AS failed_runs,
    AVG(r.current_loss)
        FILTER (WHERE r.state = 'COMPLETED') AS average_final_loss,
    MAX(r.current_epoch) AS maximum_epoch
FROM experiments e
LEFT JOIN training_runs r
    ON r.experiment_id = e.experiment_id
GROUP BY e.experiment_id, e.experiment_name;

-- ------------------------------------------------------------
-- 14. Demonstration experiment
-- ------------------------------------------------------------

INSERT INTO experiments (
    experiment_name,
    description,
    input_dimensions,
    learning_rate,
    gradient_clip
)
VALUES (
    'XOR_BACKPROP_DEMO',
    'Two-layer neural network demonstrating reverse-mode differentiation.',
    2,
    0.08,
    5.0
);

INSERT INTO computational_graphs (
    experiment_id,
    graph_name,
    graph_version,
    is_acyclic
)
SELECT
    experiment_id,
    'CHAIN_RULE_GRAPH',
    1,
    TRUE
FROM experiments
WHERE experiment_name = 'XOR_BACKPROP_DEMO';

-- f(x) = x*x + 3*x + 2
INSERT INTO graph_nodes (
    graph_id, node_name, operation,
    forward_value, requires_gradient
)
SELECT
    graph_id, 'x', 'INPUT', 2, TRUE
FROM computational_graphs
WHERE graph_name = 'CHAIN_RULE_GRAPH';

INSERT INTO graph_nodes (
    graph_id, node_name, operation, forward_value
)
SELECT
    graph_id, 'three', 'PARAMETER', 3
FROM computational_graphs
WHERE graph_name = 'CHAIN_RULE_GRAPH';

INSERT INTO graph_nodes (
    graph_id, node_name, operation, forward_value
)
SELECT
    graph_id, 'two', 'PARAMETER', 2
FROM computational_graphs
WHERE graph_name = 'CHAIN_RULE_GRAPH';

INSERT INTO graph_nodes (
    graph_id, node_name, operation, forward_value
)
SELECT
    graph_id, 'square', 'MULTIPLY', 4
FROM computational_graphs
WHERE graph_name = 'CHAIN_RULE_GRAPH';

INSERT INTO graph_nodes (
    graph_id, node_name, operation, forward_value
)
SELECT
    graph_id, 'linear', 'MULTIPLY', 6
FROM computational_graphs
WHERE graph_name = 'CHAIN_RULE_GRAPH';

INSERT INTO graph_nodes (
    graph_id, node_name, operation, forward_value
)
SELECT
    graph_id, 'partial', 'ADD', 10
FROM computational_graphs
WHERE graph_name = 'CHAIN_RULE_GRAPH';

INSERT INTO graph_nodes (
    graph_id, node_name, operation, forward_value
)
SELECT
    graph_id, 'output', 'ADD', 12
FROM computational_graphs
WHERE graph_name = 'CHAIN_RULE_GRAPH';

-- Insert graph dependencies.
INSERT INTO graph_edges (
    graph_id, parent_node_id, child_node_id, input_position
)
SELECT
    g.graph_id,
    parent.node_id,
    child.node_id,
    dependency.input_position
FROM computational_graphs g
JOIN (
    VALUES
        ('x', 'square', 0),
        ('x', 'square', 1),
        ('three', 'linear', 0),
        ('x', 'linear', 1),
        ('square', 'partial', 0),
        ('linear', 'partial', 1),
        ('partial', 'output', 0),
        ('two', 'output', 1)
) AS dependency(parent_name, child_name, input_position)
    ON TRUE
JOIN graph_nodes parent
    ON parent.graph_id = g.graph_id
    AND parent.node_name = dependency.parent_name
JOIN graph_nodes child
    ON child.graph_id = g.graph_id
    AND child.node_name = dependency.child_name
WHERE g.graph_name = 'CHAIN_RULE_GRAPH';

-- ------------------------------------------------------------
-- 15. Demonstration tensor metadata
-- ------------------------------------------------------------

INSERT INTO tensor_metadata (
    node_id, data_type, shape, element_count, device
)
SELECT
    node_id,
    'FLOAT64',
    ARRAY[1],
    1,
    'CPU'
FROM graph_nodes
WHERE node_name IN ('x', 'output')
AND graph_id = (
    SELECT graph_id
    FROM computational_graphs
    WHERE graph_name = 'CHAIN_RULE_GRAPH'
);

-- ------------------------------------------------------------
-- 16. XOR dataset
-- ------------------------------------------------------------

INSERT INTO training_datasets (
    experiment_id,
    dataset_name,
    sample_count,
    feature_count,
    dataset_hash
)
SELECT
    experiment_id,
    'XOR_TRAINING_DATA',
    4,
    2,
    'xor-demo-sha256-0001'
FROM experiments
WHERE experiment_name = 'XOR_BACKPROP_DEMO';

INSERT INTO dataset_samples (
    dataset_id, sample_number, features, target
)
SELECT dataset_id, sample_number, features, target
FROM training_datasets
CROSS JOIN (
    VALUES
        (0, ARRAY[0.0, 0.0]::NUMERIC[], 0.0::NUMERIC),
        (1, ARRAY[0.0, 1.0]::NUMERIC[], 1.0::NUMERIC),
        (2, ARRAY[1.0, 0.0]::NUMERIC[], 1.0::NUMERIC),
        (3, ARRAY[1.0, 1.0]::NUMERIC[], 0.0::NUMERIC)
) AS samples(sample_number, features, target)
WHERE dataset_name = 'XOR_TRAINING_DATA';

-- ------------------------------------------------------------
-- 17. Training run initialization
-- ------------------------------------------------------------

INSERT INTO training_runs (
    experiment_id,
    graph_id,
    state,
    total_epochs,
    started_at
)
SELECT
    e.experiment_id,
    g.graph_id,
    'CREATED',
    100,
    CURRENT_TIMESTAMP
FROM experiments e
JOIN computational_graphs g
    ON g.experiment_id = e.experiment_id
WHERE e.experiment_name = 'XOR_BACKPROP_DEMO'
AND g.graph_name = 'CHAIN_RULE_GRAPH';

-- ------------------------------------------------------------
-- 18. Workflow processing demonstration
-- ------------------------------------------------------------

DO $$
DECLARE
    selected_run BIGINT;
BEGIN
    SELECT run_id
    INTO selected_run
    FROM training_runs
    WHERE state = 'CREATED'
    ORDER BY run_id
    LIMIT 1;

    UPDATE training_runs
    SET state = 'VALIDATING'
    WHERE run_id = selected_run;

    UPDATE training_runs
    SET state = 'FORWARD'
    WHERE run_id = selected_run;

    UPDATE training_runs
    SET state = 'BACKWARD'
    WHERE run_id = selected_run;

    UPDATE training_runs
    SET state = 'UPDATING',
        current_epoch = 1,
        current_loss = 0.125
    WHERE run_id = selected_run;

    INSERT INTO optimization_steps (
        run_id,
        epoch,
        learning_rate,
        loss_before,
        loss_after,
        gradient_norm
    )
    VALUES (
        selected_run,
        1,
        0.08,
        0.250,
        0.125,
        0.750
    );

    UPDATE training_runs
    SET state = 'FORWARD'
    WHERE run_id = selected_run;

    UPDATE training_runs
    SET state = 'BACKWARD'
    WHERE run_id = selected_run;

    UPDATE training_runs
    SET state = 'UPDATING',
        current_epoch = 100,
        current_loss = 0.001
    WHERE run_id = selected_run;

    UPDATE training_runs
    SET state = 'COMPLETED',
        completed_at = CURRENT_TIMESTAMP
    WHERE run_id = selected_run;
END;
$$;

-- ------------------------------------------------------------
-- 19. Gradient audit sample
-- ------------------------------------------------------------

INSERT INTO gradient_records (
    step_id,
    node_id,
    analytical_gradient,
    numerical_gradient,
    absolute_error,
    verification_passed
)
SELECT
    step.step_id,
    node.node_id,
    7.0,
    7.000001,
    0.000001,
    TRUE
FROM optimization_steps step
JOIN training_runs run
    ON run.run_id = step.run_id
JOIN graph_nodes node
    ON node.graph_id = run.graph_id
WHERE step.epoch = 1
AND node.node_name = 'x';

-- ------------------------------------------------------------
-- 20. Gradient verification query
-- ------------------------------------------------------------

SELECT
    gradient_id,
    node_id,
    analytical_gradient,
    numerical_gradient,
    absolute_error,
    verification_passed
FROM gradient_records
ORDER BY gradient_id;

-- ------------------------------------------------------------
-- 21. Computational graph retrieval
-- ------------------------------------------------------------

SELECT
    node.node_name,
    node.operation,
    node.forward_value,
    edge.input_position,
    parent.node_name AS parent_node
FROM graph_nodes node
LEFT JOIN graph_edges edge
    ON edge.child_node_id = node.node_id
LEFT JOIN graph_nodes parent
    ON parent.node_id = edge.parent_node_id
WHERE node.graph_id = (
    SELECT graph_id
    FROM computational_graphs
    WHERE graph_name = 'CHAIN_RULE_GRAPH'
)
ORDER BY node.node_id, edge.input_position;

-- ------------------------------------------------------------
-- 22. Recursive graph traversal
-- ------------------------------------------------------------

WITH RECURSIVE graph_traversal AS (
    SELECT
        node.node_id,
        node.node_name,
        node.operation,
        0 AS depth,
        ARRAY[node.node_id] AS visited
    FROM graph_nodes node
    WHERE node.node_name = 'x'
    AND node.graph_id = (
        SELECT graph_id
        FROM computational_graphs
        WHERE graph_name = 'CHAIN_RULE_GRAPH'
    )

    UNION ALL

    SELECT
        child.node_id,
        child.node_name,
        child.operation,
        traversal.depth + 1,
        traversal.visited || child.node_id
    FROM graph_traversal traversal
    JOIN graph_edges edge
        ON edge.parent_node_id = traversal.node_id
    JOIN graph_nodes child
        ON child.node_id = edge.child_node_id
    WHERE NOT child.node_id = ANY(traversal.visited)
)
SELECT *
FROM graph_traversal
ORDER BY depth, node_id;

-- ------------------------------------------------------------
-- 23. Dataset validation
-- ------------------------------------------------------------

SELECT
    dataset.dataset_name,
    dataset.sample_count,
    COUNT(sample.sample_id) AS actual_samples,
    COUNT(sample.sample_id) = dataset.sample_count
        AS sample_count_valid,
    bool_and(
        cardinality(sample.features) = dataset.feature_count
    ) AS dimensions_valid
FROM training_datasets dataset
LEFT JOIN dataset_samples sample
    ON sample.dataset_id = dataset.dataset_id
GROUP BY dataset.dataset_id, dataset.dataset_name,
         dataset.sample_count;

-- ------------------------------------------------------------
-- 24. Training performance aggregation
-- ------------------------------------------------------------

SELECT
    run_id,
    COUNT(*) AS optimization_steps,
    MIN(loss_before) AS initial_loss,
    MIN(loss_after) AS best_loss,
    AVG(gradient_norm) AS average_gradient_norm,
    MAX(gradient_norm) AS maximum_gradient_norm,
    COUNT(*) FILTER (
        WHERE gradient_clipped
    ) AS clipped_steps
FROM optimization_steps
GROUP BY run_id;

-- ------------------------------------------------------------
-- 25. Experiment reporting
-- ------------------------------------------------------------

SELECT *
FROM experiment_training_report
ORDER BY experiment_id;

-- ------------------------------------------------------------
-- 26. Eligibility determination
--
-- A run is eligible for successful completion reporting
-- only when all epochs are completed and the final loss
-- satisfies the configured threshold.
-- ------------------------------------------------------------

SELECT
    run.run_id,
    experiment.experiment_name,
    run.state,
    run.current_epoch,
    run.total_epochs,
    run.current_loss,
    CASE
        WHEN run.state <> 'COMPLETED'
            THEN 'NOT_COMPLETED'
        WHEN run.current_epoch <> run.total_epochs
            THEN 'INCOMPLETE_EPOCHS'
        WHEN run.current_loss IS NULL
            THEN 'MISSING_LOSS'
        WHEN run.current_loss > 0.01
            THEN 'LOSS_THRESHOLD_NOT_MET'
        ELSE 'ELIGIBLE'
    END AS evaluation_status
FROM training_runs run
JOIN experiments experiment
    ON experiment.experiment_id = run.experiment_id;

-- ------------------------------------------------------------
-- 27. Demonstrate failure validation in an isolated block
-- ------------------------------------------------------------

DO $$
DECLARE
    test_result BOOLEAN := FALSE;
BEGIN
    BEGIN
        PERFORM calculate_node(
            'LOG',
            ARRAY[-1.0]::NUMERIC[]
        );
    EXCEPTION
        WHEN OTHERS THEN
            test_result := TRUE;
            RAISE NOTICE 'Expected invalid LOG input rejected: %',
                SQLERRM;
    END;

    IF NOT test_result THEN
        RAISE EXCEPTION 'Invalid LOG input was not rejected';
    END IF;
END;
$$;

COMMIT;
