-- ============================================================
-- OPTIMIZER LABORATORY
-- PostgreSQL 15+
--
-- Relational model for optimizer experiments, configurations,
-- training runs, parameter checkpoints, and convergence metrics.
--
-- Algorithms:
-- Momentum, AdaGrad, RMSProp, Adam, AdamW.
--
-- Execute the script in a database where the executing role
-- has permission to create schemas, tables, functions, and triggers.
-- ============================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS optimizer_lab;

SET search_path TO optimizer_lab, public;

-- ------------------------------------------------------------
-- 1. Domain types
-- ------------------------------------------------------------

CREATE TYPE optimizer_type AS ENUM (
    'MOMENTUM',
    'ADAGRAD',
    'RMSPROP',
    'ADAM',
    'ADAMW'
);

CREATE TYPE training_status AS ENUM (
    'CREATED',
    'QUEUED',
    'RUNNING',
    'COMPLETED',
    'FAILED',
    'CANCELLED'
);

CREATE TYPE experiment_status AS ENUM (
    'DRAFT',
    'ACTIVE',
    'ARCHIVED'
);

CREATE TYPE event_type AS ENUM (
    'CREATED',
    'STARTED',
    'CHECKPOINT',
    'CONVERGED',
    'FAILED',
    'CANCELLED'
);

-- ------------------------------------------------------------
-- 2. Experiment ownership and model definitions
-- ------------------------------------------------------------

CREATE TABLE experiment_owner (
    owner_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    owner_name VARCHAR(120) NOT NULL,
    email VARCHAR(254) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT owner_name_not_empty
        CHECK (length(trim(owner_name)) > 0)
);

CREATE TABLE experiment (
    experiment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    owner_id BIGINT NOT NULL REFERENCES experiment_owner(owner_id),
    experiment_name VARCHAR(160) NOT NULL,
    description TEXT,
    status experiment_status NOT NULL DEFAULT 'DRAFT',
    objective_dimensions INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT positive_dimensions
        CHECK (objective_dimensions > 0),

    CONSTRAINT experiment_name_not_empty
        CHECK (length(trim(experiment_name)) > 0),

    UNIQUE (owner_id, experiment_name)
);

CREATE TABLE optimizer_configuration (
    configuration_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id BIGINT NOT NULL
        REFERENCES experiment(experiment_id),

    optimizer optimizer_type NOT NULL,
    learning_rate DOUBLE PRECISION NOT NULL,
    epsilon DOUBLE PRECISION NOT NULL DEFAULT 0.00000001,
    beta1 DOUBLE PRECISION NOT NULL DEFAULT 0.9,
    beta2 DOUBLE PRECISION NOT NULL DEFAULT 0.999,
    weight_decay DOUBLE PRECISION NOT NULL DEFAULT 0,
    gradient_clip DOUBLE PRECISION,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT positive_learning_rate
        CHECK (
            learning_rate > 0
            AND learning_rate < 'Infinity'::DOUBLE PRECISION
        ),

    CONSTRAINT positive_epsilon
        CHECK (
            epsilon > 0
            AND epsilon < 'Infinity'::DOUBLE PRECISION
        ),

    CONSTRAINT valid_beta1
        CHECK (beta1 >= 0 AND beta1 < 1),

    CONSTRAINT valid_beta2
        CHECK (beta2 >= 0 AND beta2 < 1),

    CONSTRAINT nonnegative_weight_decay
        CHECK (
            weight_decay >= 0
            AND weight_decay < 'Infinity'::DOUBLE PRECISION
        ),

    CONSTRAINT valid_gradient_clip
        CHECK (
            gradient_clip IS NULL OR gradient_clip > 0
        ),

    CONSTRAINT optimizer_configuration_identity
        UNIQUE (experiment_id, configuration_id),

    CONSTRAINT adamw_decay_policy
        CHECK (
            optimizer <> 'ADAMW'
            OR weight_decay >= 0
        )
);

-- ------------------------------------------------------------
-- 3. Training jobs and parameter checkpoints
-- ------------------------------------------------------------

CREATE TABLE training_run (
    run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    experiment_id BIGINT NOT NULL
        REFERENCES experiment(experiment_id),

    configuration_id BIGINT NOT NULL,

    run_name VARCHAR(160) NOT NULL,
    status training_status NOT NULL DEFAULT 'CREATED',

    maximum_iterations INTEGER NOT NULL,
    current_iteration INTEGER NOT NULL DEFAULT 0,

    initial_loss DOUBLE PRECISION,
    final_loss DOUBLE PRECISION,

    error_message TEXT,

    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT run_configuration_fk
        FOREIGN KEY (experiment_id, configuration_id)
        REFERENCES optimizer_configuration(
            experiment_id,
            configuration_id
        ),

    CONSTRAINT positive_maximum_iterations
        CHECK (maximum_iterations > 0),

    CONSTRAINT valid_iteration
        CHECK (
            current_iteration >= 0
            AND current_iteration <= maximum_iterations
        ),

    CONSTRAINT nonnegative_initial_loss
        CHECK (
            initial_loss IS NULL OR
            (
                initial_loss >= 0
                AND initial_loss < 'Infinity'::DOUBLE PRECISION
            )
        ),

    CONSTRAINT nonnegative_final_loss
        CHECK (
            final_loss IS NULL OR
            (
                final_loss >= 0
                AND final_loss < 'Infinity'::DOUBLE PRECISION
            )
        ),

    CONSTRAINT completed_run_has_timestamp
        CHECK (
            status <> 'COMPLETED'
            OR completed_at IS NOT NULL
        ),

    UNIQUE (experiment_id, run_name)
);

CREATE TABLE parameter_checkpoint (
    checkpoint_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES training_run(run_id)
        ON DELETE CASCADE,

    iteration INTEGER NOT NULL,
    parameters DOUBLE PRECISION[] NOT NULL,
    gradient_norm DOUBLE PRECISION,
    loss DOUBLE PRECISION NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT positive_iteration
        CHECK (iteration >= 0),

    CONSTRAINT valid_checkpoint_loss
        CHECK (
            loss >= 0
            AND loss < 'Infinity'::DOUBLE PRECISION
        ),

    CONSTRAINT valid_gradient_norm
        CHECK (
            gradient_norm IS NULL OR
            (
                gradient_norm >= 0
                AND gradient_norm < 'Infinity'::DOUBLE PRECISION
            )
        ),

    CONSTRAINT nonempty_parameters
        CHECK (cardinality(parameters) > 0),

    CONSTRAINT unique_run_iteration
        UNIQUE (run_id, iteration)
);

-- ------------------------------------------------------------
-- 4. Training metrics and lifecycle audit
-- ------------------------------------------------------------

CREATE TABLE training_metric (
    metric_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES training_run(run_id)
        ON DELETE CASCADE,

    iteration INTEGER NOT NULL,
    training_loss DOUBLE PRECISION NOT NULL,
    learning_rate DOUBLE PRECISION NOT NULL,
    gradient_norm DOUBLE PRECISION,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_metric_iteration
        CHECK (iteration >= 0),

    CONSTRAINT valid_training_loss
        CHECK (
            training_loss >= 0
            AND training_loss < 'Infinity'::DOUBLE PRECISION
        ),

    CONSTRAINT valid_metric_learning_rate
        CHECK (
            learning_rate > 0
            AND learning_rate < 'Infinity'::DOUBLE PRECISION
        ),

    CONSTRAINT valid_metric_gradient_norm
        CHECK (
            gradient_norm IS NULL OR
            (
                gradient_norm >= 0
                AND gradient_norm < 'Infinity'::DOUBLE PRECISION
            )
        ),

    UNIQUE (run_id, iteration)
);

CREATE TABLE training_event (
    event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES training_run(run_id)
        ON DELETE CASCADE,

    event event_type NOT NULL,
    event_details JSONB NOT NULL DEFAULT '{}'::JSONB,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------
-- 5. Performance-oriented indexes
-- ------------------------------------------------------------

CREATE INDEX idx_experiment_owner_status
    ON experiment(owner_id, status);

CREATE INDEX idx_configuration_optimizer
    ON optimizer_configuration(optimizer, experiment_id);

CREATE INDEX idx_run_status_created
    ON training_run(status, created_at DESC);

CREATE INDEX idx_run_experiment
    ON training_run(experiment_id, created_at DESC);

CREATE INDEX idx_checkpoint_run_iteration
    ON parameter_checkpoint(run_id, iteration DESC);

CREATE INDEX idx_metric_run_iteration
    ON training_metric(run_id, iteration DESC);

CREATE INDEX idx_training_event_run_time
    ON training_event(run_id, occurred_at DESC);

CREATE INDEX idx_training_event_details
    ON training_event USING GIN(event_details);

-- ------------------------------------------------------------
-- 6. Validation functions
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION validate_parameter_dimensions()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    expected_dimensions INTEGER;
BEGIN
    SELECT e.objective_dimensions
    INTO expected_dimensions
    FROM training_run tr
    JOIN experiment e
      ON e.experiment_id = tr.experiment_id
    WHERE tr.run_id = NEW.run_id;

    IF expected_dimensions IS NULL THEN
        RAISE EXCEPTION 'Training run does not exist.';
    END IF;

    IF cardinality(NEW.parameters) <> expected_dimensions THEN
        RAISE EXCEPTION
            'Expected % parameters, received %.',
            expected_dimensions,
            cardinality(NEW.parameters);
    END IF;

    IF EXISTS (
        SELECT 1
        FROM unnest(NEW.parameters) AS value
        WHERE value IS NULL
           OR value = 'Infinity'::DOUBLE PRECISION
           OR value = '-Infinity'::DOUBLE PRECISION
           OR value = 'NaN'::DOUBLE PRECISION
    ) THEN
        RAISE EXCEPTION 'Parameter array contains invalid values.';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_checkpoint_dimensions
BEFORE INSERT OR UPDATE ON parameter_checkpoint
FOR EACH ROW
EXECUTE FUNCTION validate_parameter_dimensions();

-- ------------------------------------------------------------
-- 7. Training lifecycle enforcement
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION enforce_training_transitions()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.status IN ('COMPLETED', 'FAILED', 'CANCELLED')
       AND NEW.status <> OLD.status THEN
        RAISE EXCEPTION
            'Terminal training state % cannot be changed to %.',
            OLD.status,
            NEW.status;
    END IF;

    IF NEW.status = 'RUNNING'
       AND OLD.status NOT IN ('CREATED', 'QUEUED') THEN
        RAISE EXCEPTION 'Invalid transition to RUNNING.';
    END IF;

    IF NEW.status = 'COMPLETED'
       AND OLD.status <> 'RUNNING' THEN
        RAISE EXCEPTION 'Only running jobs can complete.';
    END IF;

    IF NEW.status = 'FAILED'
       AND OLD.status NOT IN ('RUNNING', 'CREATED', 'QUEUED') THEN
        RAISE EXCEPTION 'Invalid transition to FAILED.';
    END IF;

    IF NEW.status = 'CANCELLED'
       AND OLD.status NOT IN ('RUNNING', 'CREATED', 'QUEUED') THEN
        RAISE EXCEPTION 'Invalid transition to CANCELLED.';
    END IF;

    IF NEW.status = 'RUNNING' AND OLD.status <> 'RUNNING' THEN
        NEW.started_at := CURRENT_TIMESTAMP;
    END IF;

    IF NEW.status IN ('COMPLETED', 'FAILED', 'CANCELLED') THEN
        NEW.completed_at := CURRENT_TIMESTAMP;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_training_state
BEFORE UPDATE OF status ON training_run
FOR EACH ROW
EXECUTE FUNCTION enforce_training_transitions();

-- ------------------------------------------------------------
-- 8. Automatic lifecycle audit
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION audit_training_status()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    selected_event event_type;
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO training_event(run_id, event, event_details)
        VALUES (
            NEW.run_id,
            'CREATED',
            jsonb_build_object('status', NEW.status)
        );

        RETURN NEW;
    END IF;

    IF NEW.status IS DISTINCT FROM OLD.status THEN
        selected_event := CASE NEW.status
            WHEN 'RUNNING' THEN 'STARTED'::event_type
            WHEN 'COMPLETED' THEN 'CONVERGED'::event_type
            WHEN 'FAILED' THEN 'FAILED'::event_type
            WHEN 'CANCELLED' THEN 'CANCELLED'::event_type
            ELSE 'CHECKPOINT'::event_type
        END;

        INSERT INTO training_event(run_id, event, event_details)
        VALUES (
            NEW.run_id,
            selected_event,
            jsonb_build_object(
                'previous_status', OLD.status,
                'new_status', NEW.status,
                'iteration', NEW.current_iteration
            )
        );
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_training_audit_insert
AFTER INSERT ON training_run
FOR EACH ROW
EXECUTE FUNCTION audit_training_status();

CREATE TRIGGER trg_training_audit_update
AFTER UPDATE OF status ON training_run
FOR EACH ROW
EXECUTE FUNCTION audit_training_status();

-- ------------------------------------------------------------
-- 9. Optimizer configuration evaluation
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION optimizer_configuration_is_valid(
    selected_configuration BIGINT
)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
AS $$
    SELECT EXISTS (
        SELECT 1
        FROM optimizer_configuration oc
        JOIN experiment e
          ON e.experiment_id = oc.experiment_id
        WHERE oc.configuration_id = selected_configuration
          AND oc.learning_rate > 0
          AND oc.epsilon > 0
          AND oc.beta1 >= 0
          AND oc.beta1 < 1
          AND oc.beta2 >= 0
          AND oc.beta2 < 1
          AND (
              oc.optimizer <> 'ADAMW'
              OR oc.weight_decay >= 0
          )
          AND e.status <> 'ARCHIVED'
    );
$$;

-- ------------------------------------------------------------
-- 10. Job creation procedure
-- ------------------------------------------------------------

CREATE OR REPLACE PROCEDURE create_training_run(
    selected_experiment BIGINT,
    selected_configuration BIGINT,
    selected_name VARCHAR(160),
    selected_maximum_iterations INTEGER
)
LANGUAGE plpgsql
AS $$
BEGIN
    IF selected_maximum_iterations <= 0 THEN
        RAISE EXCEPTION 'Maximum iterations must be positive.';
    END IF;

    IF NOT optimizer_configuration_is_valid(selected_configuration) THEN
        RAISE EXCEPTION 'Optimizer configuration is not eligible.';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM optimizer_configuration
        WHERE configuration_id = selected_configuration
          AND experiment_id = selected_experiment
    ) THEN
        RAISE EXCEPTION
            'Configuration does not belong to selected experiment.';
    END IF;

    INSERT INTO training_run (
        experiment_id,
        configuration_id,
        run_name,
        maximum_iterations
    )
    VALUES (
        selected_experiment,
        selected_configuration,
        selected_name,
        selected_maximum_iterations
    );
END;
$$;

-- ------------------------------------------------------------
-- 11. Checkpoint recording procedure
-- ------------------------------------------------------------

CREATE OR REPLACE PROCEDURE record_checkpoint(
    selected_run BIGINT,
    selected_iteration INTEGER,
    selected_parameters DOUBLE PRECISION[],
    selected_gradient_norm DOUBLE PRECISION,
    selected_loss DOUBLE PRECISION
)
LANGUAGE plpgsql
AS $$
DECLARE
    current_status training_status;
    expected_next_iteration INTEGER;
    selected_learning_rate DOUBLE PRECISION;
BEGIN
    SELECT tr.status, tr.current_iteration, oc.learning_rate
    INTO current_status, expected_next_iteration, selected_learning_rate
    FROM training_run tr
    JOIN optimizer_configuration oc
      ON oc.configuration_id = tr.configuration_id
    WHERE tr.run_id = selected_run
    FOR UPDATE OF tr;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Training run not found.';
    END IF;

    IF current_status <> 'RUNNING' THEN
        RAISE EXCEPTION 'Only running jobs accept checkpoints.';
    END IF;

    IF selected_iteration <= expected_next_iteration THEN
        RAISE EXCEPTION
            'Checkpoint iteration must advance beyond current iteration.';
    END IF;

    IF selected_loss < 0 OR selected_loss = 'NaN'::DOUBLE PRECISION
       OR selected_loss = 'Infinity'::DOUBLE PRECISION THEN
        RAISE EXCEPTION 'Invalid checkpoint loss.';
    END IF;

    INSERT INTO parameter_checkpoint (
        run_id,
        iteration,
        parameters,
        gradient_norm,
        loss
    )
    VALUES (
        selected_run,
        selected_iteration,
        selected_parameters,
        selected_gradient_norm,
        selected_loss
    );

    INSERT INTO training_metric (
        run_id,
        iteration,
        training_loss,
        learning_rate,
        gradient_norm
    )
    VALUES (
        selected_run,
        selected_iteration,
        selected_loss,
        selected_learning_rate,
        selected_gradient_norm
    );

    UPDATE training_run
    SET current_iteration = selected_iteration,
        final_loss = selected_loss
    WHERE run_id = selected_run;
END;
$$;

-- ------------------------------------------------------------
-- 12. Eligibility and reporting views
-- ------------------------------------------------------------

CREATE OR REPLACE VIEW optimizer_run_report AS
SELECT
    tr.run_id,
    e.experiment_name,
    eo.owner_name,
    tr.run_name,
    oc.optimizer,
    oc.learning_rate,
    oc.weight_decay,
    tr.status,
    tr.current_iteration,
    tr.maximum_iterations,
    tr.initial_loss,
    tr.final_loss,
    tr.created_at,
    tr.completed_at
FROM training_run tr
JOIN experiment e
  ON e.experiment_id = tr.experiment_id
JOIN experiment_owner eo
  ON eo.owner_id = e.owner_id
JOIN optimizer_configuration oc
  ON oc.configuration_id = tr.configuration_id;

CREATE OR REPLACE VIEW optimizer_convergence_report AS
SELECT
    oc.optimizer,
    COUNT(tr.run_id) AS total_runs,
    COUNT(*) FILTER (
        WHERE tr.status = 'COMPLETED'
    ) AS completed_runs,
    COUNT(*) FILTER (
        WHERE tr.status = 'FAILED'
    ) AS failed_runs,
    AVG(tr.initial_loss) AS average_initial_loss,
    AVG(tr.final_loss) AS average_final_loss,
    AVG(tr.current_iteration) AS average_iterations
FROM training_run tr
JOIN optimizer_configuration oc
  ON oc.configuration_id = tr.configuration_id
GROUP BY oc.optimizer;

-- ------------------------------------------------------------
-- 13. Sample records
-- ------------------------------------------------------------

INSERT INTO experiment_owner(owner_name, email)
VALUES
    ('Research Engineering', 'research@example.org'),
    ('Applied Optimization Team', 'optimization@example.org');

INSERT INTO experiment (
    owner_id,
    experiment_name,
    description,
    status,
    objective_dimensions
)
VALUES
    (
        1,
        'Convex Optimization Benchmark',
        'Comparison of five first-order optimization algorithms.',
        'ACTIVE',
        3
    ),
    (
        2,
        'Regularization Evaluation',
        'Evaluation of decoupled weight decay.',
        'ACTIVE',
        3
    ),
    (
        1,
        'Archived Experiment',
        'Historical experiment retained for reporting.',
        'ARCHIVED',
        2
    );

INSERT INTO optimizer_configuration (
    experiment_id,
    optimizer,
    learning_rate,
    epsilon,
    beta1,
    beta2,
    weight_decay,
    gradient_clip
)
VALUES
    (1, 'MOMENTUM', 0.005, 1e-8, 0.9, 0.999, 0, 10),
    (1, 'ADAGRAD', 0.8, 1e-8, 0.9, 0.999, 0, 10),
    (1, 'RMSPROP', 0.02, 1e-8, 0.9, 0.999, 0, 10),
    (1, 'ADAM', 0.08, 1e-8, 0.9, 0.999, 0, 10),
    (2, 'ADAMW', 0.08, 1e-8, 0.9, 0.999, 0.001, 10);

-- ------------------------------------------------------------
-- 14. Create training runs
-- ------------------------------------------------------------

CALL create_training_run(1, 1, 'momentum-baseline', 3000);
CALL create_training_run(1, 2, 'adagrad-baseline', 3000);
CALL create_training_run(1, 3, 'rmsprop-baseline', 3000);
CALL create_training_run(1, 4, 'adam-baseline', 3000);
CALL create_training_run(2, 5, 'adamw-regularization', 3000);

-- Deliberately create a queued job to represent a pending state.
UPDATE training_run
SET status = 'QUEUED'
WHERE run_name = 'adamw-regularization';

-- ------------------------------------------------------------
-- 15. Start four jobs
-- ------------------------------------------------------------

UPDATE training_run
SET status = 'RUNNING',
    initial_loss = 1250.0
WHERE status = 'CREATED';

-- ------------------------------------------------------------
-- 16. Record representative checkpoints
-- ------------------------------------------------------------

CALL record_checkpoint(
    1, 10,
    ARRAY[9.2, 7.1, -4.2]::DOUBLE PRECISION[],
    12.5,
    980.0
);

CALL record_checkpoint(
    2, 10,
    ARRAY[8.5, 6.4, -3.7]::DOUBLE PRECISION[],
    8.5,
    750.0
);

CALL record_checkpoint(
    3, 10,
    ARRAY[7.9, 5.8, -3.0]::DOUBLE PRECISION[],
    7.2,
    620.0
);

CALL record_checkpoint(
    4, 10,
    ARRAY[7.2, 4.5, -2.8]::DOUBLE PRECISION[],
    6.1,
    410.0
);

-- ------------------------------------------------------------
-- 17. Complete one training run
-- ------------------------------------------------------------

UPDATE training_run
SET status = 'COMPLETED'
WHERE run_id = 4;

-- ------------------------------------------------------------
-- 18. Demonstrate failure-state recording
-- ------------------------------------------------------------

INSERT INTO training_run (
    experiment_id,
    configuration_id,
    run_name,
    maximum_iterations,
    status,
    error_message
)
VALUES (
    1,
    4,
    'adam-invalid-input-demonstration',
    100,
    'CREATED',
    NULL
);

UPDATE training_run
SET status = 'RUNNING',
    initial_loss = 100.0
WHERE run_name = 'adam-invalid-input-demonstration';

UPDATE training_run
SET status = 'FAILED',
    error_message = 'Non-finite gradient detected during numerical validation.'
WHERE run_name = 'adam-invalid-input-demonstration';

-- ------------------------------------------------------------
-- 19. Retrieval queries
-- ------------------------------------------------------------

SELECT *
FROM optimizer_run_report
ORDER BY run_id;

SELECT
    run_id,
    iteration,
    loss,
    gradient_norm,
    parameters
FROM parameter_checkpoint
ORDER BY run_id, iteration;

-- ------------------------------------------------------------
-- 20. Configuration validation
-- ------------------------------------------------------------

SELECT
    configuration_id,
    optimizer,
    optimizer_configuration_is_valid(configuration_id)
        AS eligible_configuration
FROM optimizer_configuration
ORDER BY configuration_id;

-- ------------------------------------------------------------
-- 21. Convergence aggregation
-- ------------------------------------------------------------

SELECT *
FROM optimizer_convergence_report
ORDER BY optimizer;

-- ------------------------------------------------------------
-- 22. Loss history analysis
-- ------------------------------------------------------------

WITH first_last AS (
    SELECT
        run_id,
        MIN(loss) FILTER (
            WHERE iteration = 0
        ) AS starting_loss,
        MAX(loss) FILTER (
            WHERE iteration = (
                SELECT MAX(pc2.iteration)
                FROM parameter_checkpoint pc2
                WHERE pc2.run_id = pc.run_id
            )
        ) AS latest_checkpoint_loss
    FROM parameter_checkpoint pc
    GROUP BY run_id
)
SELECT
    run_id,
    starting_loss,
    latest_checkpoint_loss
FROM first_last
ORDER BY run_id;

-- ------------------------------------------------------------
-- 23. Workflow monitoring
-- ------------------------------------------------------------

SELECT
    status,
    COUNT(*) AS job_count
FROM training_run
GROUP BY status
ORDER BY status;

SELECT
    event,
    COUNT(*) AS event_count
FROM training_event
GROUP BY event
ORDER BY event;

-- ------------------------------------------------------------
-- 24. Failure analysis
-- ------------------------------------------------------------

SELECT
    run_id,
    run_name,
    status,
    error_message,
    completed_at
FROM training_run
WHERE status = 'FAILED'
ORDER BY completed_at DESC;

-- ------------------------------------------------------------
-- 25. Ranking optimizer configurations by observed loss
-- ------------------------------------------------------------

WITH ranked_runs AS (
    SELECT
        oc.optimizer,
        tr.run_id,
        tr.final_loss,
        RANK() OVER (
            PARTITION BY tr.experiment_id
            ORDER BY tr.final_loss ASC NULLS LAST
        ) AS experiment_rank
    FROM training_run tr
    JOIN optimizer_configuration oc
      ON oc.configuration_id = tr.configuration_id
    WHERE tr.status = 'COMPLETED'
)
SELECT
    optimizer,
    run_id,
    final_loss,
    experiment_rank
FROM ranked_runs
ORDER BY experiment_rank, optimizer;

-- ------------------------------------------------------------
-- 26. Transactional cancellation example
-- ------------------------------------------------------------

SAVEPOINT before_cancellation;

UPDATE training_run
SET status = 'CANCELLED'
WHERE run_name = 'adamw-regularization'
  AND status = 'QUEUED';

-- Preserve the cancellation as part of the workflow.
RELEASE SAVEPOINT before_cancellation;

COMMIT;
