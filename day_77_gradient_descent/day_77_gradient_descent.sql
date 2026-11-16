/*
===============================================================================
GRADIENT DESCENT: BATCH, STOCHASTIC AND MINI-BATCH OPTIMIZATION
PostgreSQL 15+

Purpose:
    Model an auditable gradient descent experimentation environment.

Implements:
    1. Batch gradient descent
    2. Stochastic gradient descent
    3. Mini-batch gradient descent
    4. Training datasets and samples
    5. Model parameters and optimization runs
    6. Epoch metrics and convergence records
    7. Learning-rate and batch-size validation
    8. Evaluation metrics
    9. Training lifecycle enforcement
   10. Reporting and aggregation

The SQL implementation executes a linear regression optimization
using PL/pgSQL and stores its training history relationally.

Loss function:
    J(w,b) = (1 / 2m) * SUM((prediction - actual)^2)
             + (lambda / 2) * SUM(w_i^2)

The bias is excluded from L2 regularization.
===============================================================================
*/

BEGIN;

CREATE SCHEMA IF NOT EXISTS gradient_lab;

SET search_path TO gradient_lab, public;

-- ---------------------------------------------------------------------------
-- ENUMERATED DOMAIN TYPES
-- ---------------------------------------------------------------------------

CREATE TYPE optimizer_type AS ENUM (
    'BATCH',
    'STOCHASTIC',
    'MINI_BATCH'
);

CREATE TYPE run_status AS ENUM (
    'CREATED',
    'RUNNING',
    'COMPLETED',
    'EARLY_STOPPED',
    'FAILED'
);

CREATE TYPE dataset_role AS ENUM (
    'TRAINING',
    'VALIDATION',
    'TESTING'
);

-- ---------------------------------------------------------------------------
-- DATASETS
-- ---------------------------------------------------------------------------

CREATE TABLE datasets (
    dataset_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_name VARCHAR(120) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    feature_count INTEGER NOT NULL CHECK (feature_count > 0),
    sample_count INTEGER NOT NULL DEFAULT 0 CHECK (sample_count >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE samples (
    sample_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL REFERENCES datasets(dataset_id)
        ON DELETE CASCADE,
    sample_number INTEGER NOT NULL CHECK (sample_number > 0),
    feature_values DOUBLE PRECISION[] NOT NULL,
    target_value DOUBLE PRECISION NOT NULL,
    dataset_partition dataset_role NOT NULL DEFAULT 'TRAINING',

    UNIQUE (dataset_id, sample_number),

    CHECK (cardinality(feature_values) > 0),
    CHECK (array_position(feature_values, 'NaN'::float8) IS NULL),
    CHECK (array_position(feature_values, 'Infinity'::float8) IS NULL),
    CHECK (array_position(feature_values, '-Infinity'::float8) IS NULL),
    CHECK (target_value NOT IN (
        'NaN'::float8,
        'Infinity'::float8,
        '-Infinity'::float8
    ))
);

CREATE INDEX idx_samples_dataset_partition
    ON samples(dataset_id, dataset_partition);

-- ---------------------------------------------------------------------------
-- OPTIMIZATION CONFIGURATION
-- ---------------------------------------------------------------------------

CREATE TABLE optimization_policies (
    policy_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    policy_name VARCHAR(120) NOT NULL UNIQUE,
    optimizer optimizer_type NOT NULL,
    learning_rate DOUBLE PRECISION NOT NULL
        CHECK (learning_rate > 0 AND learning_rate < 'Infinity'::float8),
    batch_size INTEGER NOT NULL DEFAULT 16 CHECK (batch_size > 0),
    regularization DOUBLE PRECISION NOT NULL DEFAULT 0
        CHECK (regularization >= 0 AND regularization < 'Infinity'::float8),
    maximum_epochs INTEGER NOT NULL CHECK (maximum_epochs > 0),
    gradient_clip DOUBLE PRECISION NOT NULL DEFAULT 10
        CHECK (gradient_clip > 0 AND gradient_clip < 'Infinity'::float8),
    convergence_tolerance DOUBLE PRECISION NOT NULL DEFAULT 0.0000000001
        CHECK (convergence_tolerance >= 0),
    patience INTEGER NOT NULL DEFAULT 15 CHECK (patience > 0),
    random_seed INTEGER NOT NULL DEFAULT 42,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------------
-- OPTIMIZATION RUNS
-- ---------------------------------------------------------------------------

CREATE TABLE optimization_runs (
    run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL REFERENCES datasets(dataset_id),
    policy_id BIGINT NOT NULL REFERENCES optimization_policies(policy_id),
    run_name VARCHAR(150) NOT NULL,
    status run_status NOT NULL DEFAULT 'CREATED',
    final_bias DOUBLE PRECISION,
    final_weights DOUBLE PRECISION[],
    initial_loss DOUBLE PRECISION,
    final_loss DOUBLE PRECISION,
    epochs_completed INTEGER NOT NULL DEFAULT 0,
    converged BOOLEAN NOT NULL DEFAULT FALSE,
    failure_reason TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (run_name),

    CHECK (epochs_completed >= 0),

    CHECK (
        (status IN ('COMPLETED', 'EARLY_STOPPED')
         AND completed_at IS NOT NULL)
        OR
        (status NOT IN ('COMPLETED', 'EARLY_STOPPED'))
    )
);

CREATE INDEX idx_runs_status_created
    ON optimization_runs(status, created_at DESC);

CREATE INDEX idx_runs_dataset
    ON optimization_runs(dataset_id, created_at DESC);

-- ---------------------------------------------------------------------------
-- MODEL PARAMETER HISTORY
-- ---------------------------------------------------------------------------

CREATE TABLE model_parameters (
    parameter_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES optimization_runs(run_id)
        ON DELETE CASCADE,
    epoch_number INTEGER NOT NULL CHECK (epoch_number >= 0),
    weights DOUBLE PRECISION[] NOT NULL,
    bias DOUBLE PRECISION NOT NULL,
    loss DOUBLE PRECISION NOT NULL CHECK (loss >= 0),
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (run_id, epoch_number)
);

-- ---------------------------------------------------------------------------
-- EPOCH METRICS
-- ---------------------------------------------------------------------------

CREATE TABLE epoch_metrics (
    metric_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES optimization_runs(run_id)
        ON DELETE CASCADE,
    epoch_number INTEGER NOT NULL CHECK (epoch_number > 0),
    training_loss DOUBLE PRECISION NOT NULL CHECK (training_loss >= 0),
    gradient_norm DOUBLE PRECISION NOT NULL CHECK (gradient_norm >= 0),
    batch_count INTEGER NOT NULL CHECK (batch_count > 0),
    samples_processed INTEGER NOT NULL CHECK (samples_processed > 0),
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (run_id, epoch_number)
);

CREATE INDEX idx_epoch_metrics_run_epoch
    ON epoch_metrics(run_id, epoch_number);

-- ---------------------------------------------------------------------------
-- MODEL EVALUATION
-- ---------------------------------------------------------------------------

CREATE TABLE model_evaluations (
    evaluation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES optimization_runs(run_id),
    dataset_id BIGINT NOT NULL REFERENCES datasets(dataset_id),
    mean_squared_error DOUBLE PRECISION NOT NULL
        CHECK (mean_squared_error >= 0),
    r_squared DOUBLE PRECISION NOT NULL,
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CHECK (r_squared BETWEEN -1e100 AND 1e100)
);

-- ---------------------------------------------------------------------------
-- SAMPLE VALIDATION
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION validate_sample_dimensions()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    expected_dimensions INTEGER;
BEGIN
    SELECT feature_count
    INTO expected_dimensions
    FROM datasets
    WHERE dataset_id = NEW.dataset_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Dataset % does not exist', NEW.dataset_id;
    END IF;

    IF cardinality(NEW.feature_values) <> expected_dimensions THEN
        RAISE EXCEPTION
            'Expected % features, received %',
            expected_dimensions,
            cardinality(NEW.feature_values);
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_sample_dimensions
BEFORE INSERT OR UPDATE ON samples
FOR EACH ROW
EXECUTE FUNCTION validate_sample_dimensions();

-- ---------------------------------------------------------------------------
-- SAMPLE COUNT MAINTENANCE
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION maintain_sample_count()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE datasets
        SET sample_count = sample_count + 1
        WHERE dataset_id = NEW.dataset_id;

        RETURN NEW;

    ELSIF TG_OP = 'DELETE' THEN
        UPDATE datasets
        SET sample_count = sample_count - 1
        WHERE dataset_id = OLD.dataset_id;

        RETURN OLD;
    END IF;

    IF OLD.dataset_id <> NEW.dataset_id THEN
        UPDATE datasets
        SET sample_count = sample_count - 1
        WHERE dataset_id = OLD.dataset_id;

        UPDATE datasets
        SET sample_count = sample_count + 1
        WHERE dataset_id = NEW.dataset_id;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_maintain_sample_count
AFTER INSERT OR DELETE OR UPDATE OF dataset_id ON samples
FOR EACH ROW
EXECUTE FUNCTION maintain_sample_count();

-- ---------------------------------------------------------------------------
-- RUN STATE TRANSITIONS
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION enforce_run_transitions()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.status IN ('COMPLETED', 'EARLY_STOPPED', 'FAILED')
       AND NEW.status <> OLD.status THEN
        RAISE EXCEPTION
            'Terminal run state % cannot transition to %',
            OLD.status,
            NEW.status;
    END IF;

    IF OLD.status = 'CREATED'
       AND NEW.status = 'RUNNING' THEN
        NEW.started_at := CURRENT_TIMESTAMP;
    END IF;

    IF NEW.status IN ('COMPLETED', 'EARLY_STOPPED', 'FAILED') THEN
        NEW.completed_at := CURRENT_TIMESTAMP;
    END IF;

    IF NEW.status = 'FAILED' AND
       NULLIF(TRIM(NEW.failure_reason), '') IS NULL THEN
        RAISE EXCEPTION 'Failed runs require a failure reason.';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_run_transitions
BEFORE UPDATE OF status ON optimization_runs
FOR EACH ROW
EXECUTE FUNCTION enforce_run_transitions();

-- ---------------------------------------------------------------------------
-- TRAINING DATASET ELIGIBILITY
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION validate_training_dataset(
    p_dataset_id BIGINT
)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    total_samples INTEGER;
    training_samples INTEGER;
BEGIN
    SELECT COUNT(*),
           COUNT(*) FILTER (
               WHERE dataset_partition = 'TRAINING'
           )
    INTO total_samples, training_samples
    FROM samples
    WHERE dataset_id = p_dataset_id;

    IF total_samples < 2 OR training_samples < 2 THEN
        RAISE EXCEPTION
            'Training dataset requires at least two training samples.';
    END IF;

    RETURN training_samples;
END;
$$;

-- ---------------------------------------------------------------------------
-- LOSS CALCULATION
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION calculate_loss(
    p_dataset_id BIGINT,
    p_weights DOUBLE PRECISION[],
    p_bias DOUBLE PRECISION,
    p_regularization DOUBLE PRECISION,
    p_partition dataset_role
)
RETURNS DOUBLE PRECISION
LANGUAGE plpgsql
AS $$
DECLARE
    sample RECORD;
    prediction DOUBLE PRECISION;
    error_value DOUBLE PRECISION;
    total_loss DOUBLE PRECISION := 0;
    penalty DOUBLE PRECISION := 0;
    sample_count INTEGER := 0;
    i INTEGER;
BEGIN
    FOR sample IN
        SELECT feature_values, target_value
        FROM samples
        WHERE dataset_id = p_dataset_id
          AND dataset_partition = p_partition
        ORDER BY sample_number
    LOOP
        prediction := p_bias;

        FOR i IN 1..cardinality(p_weights) LOOP
            prediction := prediction +
                p_weights[i] * sample.feature_values[i];
        END LOOP;

        error_value := prediction - sample.target_value;

        total_loss := total_loss + error_value * error_value;
        sample_count := sample_count + 1;
    END LOOP;

    IF sample_count = 0 THEN
        RAISE EXCEPTION 'No samples available for loss calculation.';
    END IF;

    FOR i IN 1..cardinality(p_weights) LOOP
        penalty := penalty + p_weights[i] * p_weights[i];
    END LOOP;

    RETURN total_loss / (2.0 * sample_count)
           + p_regularization * penalty / 2.0;
END;
$$;

-- ---------------------------------------------------------------------------
-- GRADIENT DESCENT TRAINING PROCEDURE
-- ---------------------------------------------------------------------------

CREATE OR REPLACE PROCEDURE train_model(
    p_run_id BIGINT
)
LANGUAGE plpgsql
AS $$
DECLARE
    run_record RECORD;
    policy_record RECORD;
    sample RECORD;

    weights DOUBLE PRECISION[];
    gradients DOUBLE PRECISION[];
    bias DOUBLE PRECISION := 0;
    bias_gradient DOUBLE PRECISION;

    current_loss DOUBLE PRECISION;
    best_loss DOUBLE PRECISION := 'Infinity'::float8;
    gradient_norm DOUBLE PRECISION;
    clipping_scale DOUBLE PRECISION;

    epoch_number INTEGER;
    batch_count INTEGER;
    batch_samples INTEGER;
    training_count INTEGER;

    stale_epochs INTEGER := 0;
    stop_training BOOLEAN := FALSE;

    current_batch_id INTEGER;
    sample_index INTEGER;
    i INTEGER;
BEGIN
    SELECT *
    INTO run_record
    FROM optimization_runs
    WHERE run_id = p_run_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Run % does not exist.', p_run_id;
    END IF;

    IF run_record.status <> 'CREATED' THEN
        RAISE EXCEPTION
            'Only CREATED runs can begin training. Current state: %',
            run_record.status;
    END IF;

    SELECT *
    INTO policy_record
    FROM optimization_policies
    WHERE policy_id = run_record.policy_id;

    training_count := validate_training_dataset(run_record.dataset_id);

    IF policy_record.optimizer = 'MINI_BATCH'
       AND policy_record.batch_size <= 0 THEN
        RAISE EXCEPTION 'Mini-batch size must be positive.';
    END IF;

    weights := array_fill(
        0.0::double precision,
        ARRAY[
            (SELECT feature_count
             FROM datasets
             WHERE dataset_id = run_record.dataset_id)
        ]
    );

    gradients := array_fill(
        0.0::double precision,
        ARRAY[cardinality(weights)]
    );

    UPDATE optimization_runs
    SET status = 'RUNNING'
    WHERE run_id = p_run_id;

    current_loss := calculate_loss(
        run_record.dataset_id,
        weights,
        bias,
        policy_record.regularization,
        'TRAINING'
    );

    UPDATE optimization_runs
    SET initial_loss = current_loss
    WHERE run_id = p_run_id;

    INSERT INTO model_parameters(
        run_id,
        epoch_number,
        weights,
        bias,
        loss
    )
    VALUES (
        p_run_id,
        0,
        weights,
        bias,
        current_loss
    );

    FOR epoch_number IN 1..policy_record.maximum_epochs LOOP

        gradients := array_fill(
            0.0::double precision,
            ARRAY[cardinality(weights)]
        );

        bias_gradient := 0;
        batch_count := 0;
        batch_samples := 0;
        gradient_norm := 0;

        /*
         * BATCH:
         * All training records contribute to one gradient update.
         *
         * STOCHASTIC:
         * Every training record contributes to an individual update.
         *
         * MINI_BATCH:
         * Training records are grouped by batch_size.
         */

        IF policy_record.optimizer = 'BATCH' THEN
            batch_count := 1;

        ELSIF policy_record.optimizer = 'STOCHASTIC' THEN
            batch_count := training_count;

        ELSE
            batch_count := CEIL(
                training_count::NUMERIC /
                policy_record.batch_size
            )::INTEGER;
        END IF;

        FOR current_batch_id IN 0..batch_count - 1 LOOP

            gradients := array_fill(
                0.0::double precision,
                ARRAY[cardinality(weights)]
            );

            bias_gradient := 0;
            batch_samples := 0;

            FOR sample IN
                SELECT
                    sample_number,
                    feature_values,
                    target_value
                FROM samples
                WHERE dataset_id = run_record.dataset_id
                  AND dataset_partition = 'TRAINING'
                ORDER BY
                    CASE
                        WHEN policy_record.optimizer = 'STOCHASTIC'
                        THEN sample_number
                        ELSE
                            ((sample_number * 1103515245::BIGINT
                              + policy_record.random_seed
                              + epoch_number * 12345::BIGINT)
                              % 2147483647)
                    END
            LOOP

                sample_index := sample.sample_number - 1;

                IF policy_record.optimizer = 'MINI_BATCH'
                   AND FLOOR(
                       sample_index::NUMERIC /
                       policy_record.batch_size
                   ) <> current_batch_id THEN
                    CONTINUE;
                END IF;

                IF policy_record.optimizer = 'STOCHASTIC'
                   AND sample_index <> current_batch_id THEN
                    CONTINUE;
                END IF;

                DECLARE
                    prediction DOUBLE PRECISION := bias;
                    error_value DOUBLE PRECISION;
                BEGIN
                    FOR i IN 1..cardinality(weights) LOOP
                        prediction := prediction +
                            weights[i] * sample.feature_values[i];
                    END LOOP;

                    error_value := prediction - sample.target_value;

                    FOR i IN 1..cardinality(weights) LOOP
                        gradients[i] := gradients[i] +
                            error_value * sample.feature_values[i];
                    END LOOP;

                    bias_gradient := bias_gradient + error_value;
                    batch_samples := batch_samples + 1;
                END;
            END LOOP;

            IF batch_samples = 0 THEN
                CONTINUE;
            END IF;

            FOR i IN 1..cardinality(weights) LOOP
                gradients[i] :=
                    gradients[i] / batch_samples
                    + policy_record.regularization * weights[i];

                gradient_norm := gradient_norm +
                    gradients[i] * gradients[i];
            END LOOP;

            bias_gradient := bias_gradient / batch_samples;

            gradient_norm := gradient_norm +
                bias_gradient * bias_gradient;

            gradient_norm := SQRT(gradient_norm);

            clipping_scale := LEAST(
                1.0,
                policy_record.gradient_clip /
                GREATEST(gradient_norm, 1e-15)
            );

            FOR i IN 1..cardinality(weights) LOOP
                weights[i] := weights[i] -
                    policy_record.learning_rate *
                    gradients[i] * clipping_scale;
            END LOOP;

            bias := bias -
                policy_record.learning_rate *
                bias_gradient * clipping_scale;

            IF NOT isfinite(bias) THEN
                RAISE EXCEPTION 'Non-finite bias at epoch %.',
                    epoch_number;
            END IF;

            FOR i IN 1..cardinality(weights) LOOP
                IF NOT isfinite(weights[i]) THEN
                    RAISE EXCEPTION
                        'Non-finite weight at epoch %.',
                        epoch_number;
                END IF;
            END LOOP;
        END LOOP;

        current_loss := calculate_loss(
            run_record.dataset_id,
            weights,
            bias,
            policy_record.regularization,
            'TRAINING'
        );

        IF NOT isfinite(current_loss) THEN
            RAISE EXCEPTION 'Non-finite loss at epoch %.',
                epoch_number;
        END IF;

        INSERT INTO epoch_metrics(
            run_id,
            epoch_number,
            training_loss,
            gradient_norm,
            batch_count,
            samples_processed
        )
        VALUES (
            p_run_id,
            epoch_number,
            current_loss,
            gradient_norm,
            batch_count,
            training_count
        );

        INSERT INTO model_parameters(
            run_id,
            epoch_number,
            weights,
            bias,
            loss
        )
        VALUES (
            p_run_id,
            epoch_number,
            weights,
            bias,
            current_loss
        );

        IF best_loss - current_loss >
           policy_record.convergence_tolerance THEN
            best_loss := current_loss;
            stale_epochs := 0;
        ELSE
            stale_epochs := stale_epochs + 1;
        END IF;

        IF stale_epochs >= policy_record.patience THEN
            stop_training := TRUE;
            EXIT;
        END IF;
    END LOOP;

    UPDATE optimization_runs
    SET status = CASE
            WHEN stop_training THEN 'EARLY_STOPPED'::run_status
            ELSE 'COMPLETED'::run_status
        END,
        final_bias = bias,
        final_weights = weights,
        final_loss = current_loss,
        epochs_completed = epoch_number,
        converged = stop_training
    WHERE run_id = p_run_id;

EXCEPTION
    WHEN OTHERS THEN
        UPDATE optimization_runs
        SET status = 'FAILED',
            failure_reason = LEFT(SQLERRM, 2000),
            completed_at = CURRENT_TIMESTAMP
        WHERE run_id = p_run_id
          AND status = 'RUNNING';

        RAISE;
END;
$$;

-- ---------------------------------------------------------------------------
-- EVALUATION PROCEDURE
-- ---------------------------------------------------------------------------

CREATE OR REPLACE PROCEDURE evaluate_model(
    p_run_id BIGINT,
    p_dataset_id BIGINT
)
LANGUAGE plpgsql
AS $$
DECLARE
    run_record RECORD;
    sample RECORD;

    prediction DOUBLE PRECISION;
    mean_target DOUBLE PRECISION;
    residual DOUBLE PRECISION := 0;
    total_sum DOUBLE PRECISION := 0;
    sample_count INTEGER := 0;
    mse DOUBLE PRECISION;
    score DOUBLE PRECISION;
    i INTEGER;
BEGIN
    SELECT *
    INTO run_record
    FROM optimization_runs
    WHERE run_id = p_run_id;

    IF NOT FOUND OR run_record.status NOT IN (
        'COMPLETED',
        'EARLY_STOPPED'
    ) THEN
        RAISE EXCEPTION 'Run is not eligible for evaluation.';
    END IF;

    IF run_record.dataset_id = p_dataset_id THEN
        RAISE EXCEPTION
            'Evaluation must use a separate dataset.';
    END IF;

    SELECT AVG(target_value)
    INTO mean_target
    FROM samples
    WHERE dataset_id = p_dataset_id
      AND dataset_partition = 'TESTING';

    IF mean_target IS NULL THEN
        RAISE EXCEPTION 'Testing samples are unavailable.';
    END IF;

    FOR sample IN
        SELECT feature_values, target_value
        FROM samples
        WHERE dataset_id = p_dataset_id
          AND dataset_partition = 'TESTING'
    LOOP
        prediction := run_record.final_bias;

        FOR i IN 1..cardinality(run_record.final_weights) LOOP
            prediction := prediction +
                run_record.final_weights[i] *
                sample.feature_values[i];
        END LOOP;

        residual := residual +
            POWER(prediction - sample.target_value, 2);

        total_sum := total_sum +
            POWER(sample.target_value - mean_target, 2);

        sample_count := sample_count + 1;
    END LOOP;

    mse := residual / sample_count;

    score := CASE
        WHEN total_sum = 0 AND residual = 0 THEN 1.0
        WHEN total_sum = 0 THEN 0.0
        ELSE 1.0 - residual / total_sum
    END;

    INSERT INTO model_evaluations(
        run_id,
        dataset_id,
        mean_squared_error,
        r_squared
    )
    VALUES (
        p_run_id,
        p_dataset_id,
        mse,
        score
    );
END;
$$;

-- ---------------------------------------------------------------------------
-- REPORTING VIEWS
-- ---------------------------------------------------------------------------

CREATE VIEW optimizer_performance AS
SELECT
    r.run_id,
    r.run_name,
    p.optimizer,
    p.learning_rate,
    p.batch_size,
    r.status,
    r.epochs_completed,
    r.initial_loss,
    r.final_loss,
    CASE
        WHEN r.initial_loss > 0
        THEN ROUND(
            (
                100 * (r.initial_loss - r.final_loss) /
                r.initial_loss
            )::NUMERIC,
            4
        )
        ELSE NULL
    END AS percentage_loss_reduction,
    r.converged,
    r.created_at
FROM optimization_runs r
JOIN optimization_policies p
    ON p.policy_id = r.policy_id;

CREATE VIEW epoch_progress AS
SELECT
    r.run_id,
    r.run_name,
    p.optimizer,
    e.epoch_number,
    e.training_loss,
    e.gradient_norm,
    e.batch_count,
    e.samples_processed
FROM epoch_metrics e
JOIN optimization_runs r
    ON r.run_id = e.run_id
JOIN optimization_policies p
    ON p.policy_id = r.policy_id;

CREATE VIEW evaluation_report AS
SELECT
    r.run_name,
    p.optimizer,
    d.dataset_name AS evaluation_dataset,
    e.mean_squared_error,
    e.r_squared,
    e.evaluated_at
FROM model_evaluations e
JOIN optimization_runs r
    ON r.run_id = e.run_id
JOIN optimization_policies p
    ON p.policy_id = r.policy_id
JOIN datasets d
    ON d.dataset_id = e.dataset_id;

-- ---------------------------------------------------------------------------
-- SAMPLE DATASETS
-- ---------------------------------------------------------------------------

INSERT INTO datasets(
    dataset_name,
    description,
    feature_count
)
VALUES
(
    'linear_training',
    'Training records for two-feature linear regression.',
    2
),
(
    'linear_testing',
    'Independent testing records for generalization evaluation.',
    2
);

-- Training data follows y = 2.5x1 - 1.7x2 + 4.
INSERT INTO samples(
    dataset_id,
    sample_number,
    feature_values,
    target_value,
    dataset_partition
)
SELECT
    1,
    i,
    ARRAY[
        ((i % 11) - 5)::DOUBLE PRECISION,
        (((i * 3) % 13) - 6)::DOUBLE PRECISION
    ],
    (
        2.5 * ((i % 11) - 5)
        - 1.7 * (((i * 3) % 13) - 6)
        + 4.0
    )::DOUBLE PRECISION,
    'TRAINING'
FROM generate_series(1, 120) AS i;

INSERT INTO samples(
    dataset_id,
    sample_number,
    feature_values,
    target_value,
    dataset_partition
)
SELECT
    2,
    i,
    ARRAY[
        ((i % 11) - 4)::DOUBLE PRECISION,
        (((i * 5) % 13) - 6)::DOUBLE PRECISION
    ],
    (
        2.5 * ((i % 11) - 4)
        - 1.7 * (((i * 5) % 13) - 6)
        + 4.0
    )::DOUBLE PRECISION,
    'TESTING'
FROM generate_series(1, 60) AS i;

-- ---------------------------------------------------------------------------
-- OPTIMIZATION POLICIES
-- ---------------------------------------------------------------------------

INSERT INTO optimization_policies(
    policy_name,
    optimizer,
    learning_rate,
    batch_size,
    regularization,
    maximum_epochs,
    gradient_clip,
    convergence_tolerance,
    patience,
    random_seed
)
VALUES
(
    'batch_standard',
    'BATCH',
    0.01,
    120,
    0.0001,
    150,
    10,
    0.0000000001,
    25,
    42
),
(
    'stochastic_standard',
    'STOCHASTIC',
    0.0005,
    1,
    0.0001,
    150,
    10,
    0.0000000001,
    25,
    42
),
(
    'mini_batch_standard',
    'MINI_BATCH',
    0.01,
    16,
    0.0001,
    150,
    10,
    0.0000000001,
    25,
    42
);

-- ---------------------------------------------------------------------------
-- CREATE OPTIMIZATION RUNS
-- ---------------------------------------------------------------------------

INSERT INTO optimization_runs(
    dataset_id,
    policy_id,
    run_name
)
VALUES
(1, 1, 'batch_linear_experiment'),
(1, 2, 'stochastic_linear_experiment'),
(1, 3, 'mini_batch_linear_experiment');

COMMIT;

-- ---------------------------------------------------------------------------
-- EXECUTE TRAINING
-- Each run is isolated in its own transaction.
-- ---------------------------------------------------------------------------

CALL gradient_lab.train_model(1);
CALL gradient_lab.train_model(2);
CALL gradient_lab.train_model(3);

-- ---------------------------------------------------------------------------
-- MODEL EVALUATION
-- ---------------------------------------------------------------------------

CALL gradient_lab.evaluate_model(1, 2);
CALL gradient_lab.evaluate_model(2, 2);
CALL gradient_lab.evaluate_model(3, 2);

-- ---------------------------------------------------------------------------
-- RETRIEVAL: OPTIMIZATION RUNS
-- ---------------------------------------------------------------------------

SELECT
    run_id,
    run_name,
    optimizer,
    status,
    epochs_completed,
    initial_loss,
    final_loss,
    percentage_loss_reduction
FROM gradient_lab.optimizer_performance
ORDER BY final_loss;

-- ---------------------------------------------------------------------------
-- RETRIEVAL: MODEL PARAMETERS
-- ---------------------------------------------------------------------------

SELECT
    r.run_name,
    p.epoch_number,
    p.weights,
    p.bias,
    p.loss
FROM gradient_lab.model_parameters p
JOIN gradient_lab.optimization_runs r
    ON r.run_id = p.run_id
ORDER BY r.run_id, p.epoch_number;

-- ---------------------------------------------------------------------------
-- AGGREGATION: MEAN EPOCH LOSS BY OPTIMIZER
-- ---------------------------------------------------------------------------

SELECT
    optimizer,
    COUNT(*) AS recorded_epochs,
    ROUND(AVG(training_loss)::NUMERIC, 6) AS average_loss,
    ROUND(MIN(training_loss)::NUMERIC, 6) AS minimum_loss,
    ROUND(MAX(training_loss)::NUMERIC, 6) AS maximum_loss,
    ROUND(AVG(gradient_norm)::NUMERIC, 6) AS average_gradient_norm
FROM gradient_lab.epoch_progress
GROUP BY optimizer
ORDER BY average_loss;

-- ---------------------------------------------------------------------------
-- EVALUATION: GENERALIZATION PERFORMANCE
-- ---------------------------------------------------------------------------

SELECT
    run_name,
    optimizer,
    mean_squared_error,
    r_squared
FROM gradient_lab.evaluation_report
ORDER BY mean_squared_error;

-- ---------------------------------------------------------------------------
-- VALIDATION: DATASET DIMENSION CONSISTENCY
-- ---------------------------------------------------------------------------

SELECT
    d.dataset_name,
    d.feature_count,
    COUNT(s.sample_id) AS actual_samples,
    COUNT(*) FILTER (
        WHERE cardinality(s.feature_values) <> d.feature_count
    ) AS invalid_dimensions
FROM gradient_lab.datasets d
LEFT JOIN gradient_lab.samples s
    ON s.dataset_id = d.dataset_id
GROUP BY d.dataset_id, d.dataset_name, d.feature_count;

-- ---------------------------------------------------------------------------
-- WORKFLOW: LATEST EPOCH FOR EACH RUN
-- ---------------------------------------------------------------------------

WITH latest_epoch AS (
    SELECT DISTINCT ON (run_id)
        run_id,
        epoch_number,
        training_loss,
        gradient_norm
    FROM gradient_lab.epoch_metrics
    ORDER BY run_id, epoch_number DESC
)
SELECT
    r.run_name,
    r.status,
    l.epoch_number,
    l.training_loss,
    l.gradient_norm
FROM gradient_lab.optimization_runs r
LEFT JOIN latest_epoch l
    ON l.run_id = r.run_id
ORDER BY r.run_id;

-- ---------------------------------------------------------------------------
-- ELIGIBILITY: COMPLETED RUNS AVAILABLE FOR EVALUATION
-- ---------------------------------------------------------------------------

SELECT
    run_id,
    run_name,
    status,
    final_loss,
    final_weights
FROM gradient_lab.optimization_runs
WHERE status IN ('COMPLETED', 'EARLY_STOPPED')
  AND final_weights IS NOT NULL;

-- ---------------------------------------------------------------------------
-- FAILURE CASE DEMONSTRATION
-- A dimensionally invalid sample is rejected by the trigger.
-- The nested block rolls back only this invalid operation.
-- ---------------------------------------------------------------------------

DO $$
BEGIN
    BEGIN
        INSERT INTO gradient_lab.samples(
            dataset_id,
            sample_number,
            feature_values,
            target_value,
            dataset_partition
        )
        VALUES (
            1,
            999,
            ARRAY[1.0, 2.0, 3.0],
            10.0,
            'TRAINING'
        );

        RAISE EXCEPTION
            'Invalid sample unexpectedly passed validation.';

    EXCEPTION
        WHEN OTHERS THEN
            IF SQLERRM LIKE 'Invalid sample unexpectedly%' THEN
                RAISE;
            END IF;

            RAISE NOTICE
                'Expected validation failure: %',
                SQLERRM;
    END;
END;
$$;

-- ---------------------------------------------------------------------------
-- PERFORMANCE DIAGNOSTICS
-- ---------------------------------------------------------------------------

EXPLAIN
SELECT
    run_id,
    epoch_number,
    training_loss
FROM gradient_lab.epoch_metrics
WHERE run_id = 1
ORDER BY epoch_number;

-- ---------------------------------------------------------------------------
-- PRODUCTION SECURITY NOTES
--
-- 1. Restrict table modification permissions for application roles.
-- 2. Expose training procedures through controlled database roles.
-- 3. Do not accept arbitrary SQL or untrusted array expressions.
-- 4. Separate training and evaluation datasets.
-- 5. Preserve audit records and restrict destructive operations.
-- 6. For large datasets, move numerical optimization into a dedicated
--    numerical service and use PostgreSQL for metadata and audit history.
-- 7. Use connection pooling and bounded training concurrency.
-- ---------------------------------------------------------------------------
