-- ============================================================
-- OBJECT DETECTION MANAGEMENT AND EVALUATION DATABASE
-- PostgreSQL 15+
--
-- Domain:
-- Images -> Models -> Detection Runs -> Predictions
--                                      -> Bounding Boxes
-- Ground Truth -> Evaluation -> Metrics
--
-- Supports IoU, confidence filtering, NMS candidate processing,
-- detection evaluation and operational reporting.
-- ============================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS object_detection;

SET search_path TO object_detection, public;

-- ------------------------------------------------------------
-- 1. ENUM TYPES
-- ------------------------------------------------------------

CREATE TYPE model_status AS ENUM (
    'DEVELOPMENT',
    'VALIDATION',
    'PRODUCTION',
    'RETIRED'
);

CREATE TYPE run_status AS ENUM (
    'QUEUED',
    'PROCESSING',
    'COMPLETED',
    'FAILED',
    'CANCELLED'
);

CREATE TYPE suppression_method AS ENUM (
    'HARD_NMS',
    'SOFT_NMS',
    'NONE'
);

CREATE TYPE evaluation_status AS ENUM (
    'PENDING',
    'RUNNING',
    'COMPLETED',
    'FAILED'
);

CREATE TYPE prediction_status AS ENUM (
    'RAW',
    'CONFIDENCE_FILTERED',
    'RETAINED',
    'SUPPRESSED'
);

-- ------------------------------------------------------------
-- 2. MODEL REGISTRY
-- ------------------------------------------------------------

CREATE TABLE detection_models (
    model_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_name VARCHAR(150) NOT NULL,
    architecture VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    framework VARCHAR(80) NOT NULL,
    input_width INTEGER NOT NULL,
    input_height INTEGER NOT NULL,
    status model_status NOT NULL DEFAULT 'DEVELOPMENT',
    model_checksum VARCHAR(128) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_model_version
        UNIQUE (model_name, model_version),

    CONSTRAINT chk_model_dimensions
        CHECK (input_width > 0 AND input_height > 0),

    CONSTRAINT chk_model_checksum
        CHECK (length(model_checksum) >= 16)
);

-- ------------------------------------------------------------
-- 3. CLASS REGISTRY
-- ------------------------------------------------------------

CREATE TABLE object_classes (
    class_id INTEGER PRIMARY KEY,
    class_name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT
);

-- ------------------------------------------------------------
-- 4. IMAGE DATASET
-- ------------------------------------------------------------

CREATE TABLE images (
    image_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    external_image_key VARCHAR(150) NOT NULL UNIQUE,
    storage_uri TEXT NOT NULL,
    image_width INTEGER NOT NULL,
    image_height INTEGER NOT NULL,
    dataset_name VARCHAR(150) NOT NULL,
    image_checksum VARCHAR(128) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_image_dimensions
        CHECK (image_width > 0 AND image_height > 0)
);

-- ------------------------------------------------------------
-- 5. GROUND-TRUTH BOUNDING BOXES
-- ------------------------------------------------------------

CREATE TABLE ground_truth_boxes (
    ground_truth_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    image_id BIGINT NOT NULL REFERENCES images(image_id)
        ON DELETE CASCADE,
    class_id INTEGER NOT NULL REFERENCES object_classes(class_id),
    x1 DOUBLE PRECISION NOT NULL,
    y1 DOUBLE PRECISION NOT NULL,
    x2 DOUBLE PRECISION NOT NULL,
    y2 DOUBLE PRECISION NOT NULL,
    difficult BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_ground_truth_coordinates
        CHECK (
            x1 >= 0 AND y1 >= 0
            AND x2 > x1 AND y2 > y1
        ),

    CONSTRAINT chk_ground_truth_finite
        CHECK (
            x1 NOT IN ('Infinity'::FLOAT8, '-Infinity'::FLOAT8,
                       'NaN'::FLOAT8)
            AND y1 NOT IN ('Infinity'::FLOAT8, '-Infinity'::FLOAT8,
                           'NaN'::FLOAT8)
            AND x2 NOT IN ('Infinity'::FLOAT8, '-Infinity'::FLOAT8,
                           'NaN'::FLOAT8)
            AND y2 NOT IN ('Infinity'::FLOAT8, '-Infinity'::FLOAT8,
                           'NaN'::FLOAT8)
        )
);

-- ------------------------------------------------------------
-- 6. DETECTION RUNS
-- ------------------------------------------------------------

CREATE TABLE detection_runs (
    run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_id BIGINT NOT NULL REFERENCES detection_models(model_id),
    image_id BIGINT NOT NULL REFERENCES images(image_id),
    status run_status NOT NULL DEFAULT 'QUEUED',
    confidence_threshold DOUBLE PRECISION NOT NULL DEFAULT 0.25,
    iou_threshold DOUBLE PRECISION NOT NULL DEFAULT 0.45,
    suppression suppression_method NOT NULL DEFAULT 'HARD_NMS',
    class_agnostic BOOLEAN NOT NULL DEFAULT FALSE,
    max_detections INTEGER NOT NULL DEFAULT 100,
    processing_started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_run_confidence
        CHECK (
            confidence_threshold BETWEEN 0 AND 1
            AND confidence_threshold NOT IN (
                'NaN'::FLOAT8,
                'Infinity'::FLOAT8,
                '-Infinity'::FLOAT8
            )
        ),

    CONSTRAINT chk_run_iou
        CHECK (
            iou_threshold BETWEEN 0 AND 1
            AND iou_threshold NOT IN (
                'NaN'::FLOAT8,
                'Infinity'::FLOAT8,
                '-Infinity'::FLOAT8
            )
        ),

    CONSTRAINT chk_max_detections
        CHECK (max_detections > 0),

    CONSTRAINT chk_run_timestamps
        CHECK (
            completed_at IS NULL
            OR processing_started_at IS NULL
            OR completed_at >= processing_started_at
        )
);

-- ------------------------------------------------------------
-- 7. PREDICTIONS
-- ------------------------------------------------------------

CREATE TABLE predictions (
    prediction_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES detection_runs(run_id)
        ON DELETE CASCADE,
    class_id INTEGER NOT NULL REFERENCES object_classes(class_id),
    x1 DOUBLE PRECISION NOT NULL,
    y1 DOUBLE PRECISION NOT NULL,
    x2 DOUBLE PRECISION NOT NULL,
    y2 DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    status prediction_status NOT NULL DEFAULT 'RAW',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_prediction_coordinates
        CHECK (
            x1 >= 0 AND y1 >= 0
            AND x2 > x1 AND y2 > y1
        ),

    CONSTRAINT chk_prediction_confidence
        CHECK (
            confidence BETWEEN 0 AND 1
            AND confidence NOT IN (
                'NaN'::FLOAT8,
                'Infinity'::FLOAT8,
                '-Infinity'::FLOAT8
            )
        ),

    CONSTRAINT chk_prediction_finite
        CHECK (
            x1 NOT IN ('Infinity'::FLOAT8, '-Infinity'::FLOAT8,
                       'NaN'::FLOAT8)
            AND y1 NOT IN ('Infinity'::FLOAT8, '-Infinity'::FLOAT8,
                           'NaN'::FLOAT8)
            AND x2 NOT IN ('Infinity'::FLOAT8, '-Infinity'::FLOAT8,
                           'NaN'::FLOAT8)
            AND y2 NOT IN ('Infinity'::FLOAT8, '-Infinity'::FLOAT8,
                           'NaN'::FLOAT8)
        )
);

-- ------------------------------------------------------------
-- 8. SUPPRESSION AUDIT
-- ------------------------------------------------------------

CREATE TABLE suppression_events (
    event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES detection_runs(run_id)
        ON DELETE CASCADE,
    selected_prediction_id BIGINT NOT NULL
        REFERENCES predictions(prediction_id),
    suppressed_prediction_id BIGINT NOT NULL
        REFERENCES predictions(prediction_id),
    overlap_iou DOUBLE PRECISION NOT NULL,
    suppression_method suppression_method NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_suppression_iou
        CHECK (overlap_iou BETWEEN 0 AND 1),

    CONSTRAINT chk_distinct_predictions
        CHECK (
            selected_prediction_id <> suppressed_prediction_id
        )
);

-- ------------------------------------------------------------
-- 9. EVALUATION RECORDS
-- ------------------------------------------------------------

CREATE TABLE evaluations (
    evaluation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES detection_runs(run_id),
    status evaluation_status NOT NULL DEFAULT 'PENDING',
    evaluation_iou_threshold DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    true_positives INTEGER NOT NULL DEFAULT 0,
    false_positives INTEGER NOT NULL DEFAULT 0,
    false_negatives INTEGER NOT NULL DEFAULT 0,
    precision_value DOUBLE PRECISION,
    recall_value DOUBLE PRECISION,
    f1_value DOUBLE PRECISION,
    average_precision DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    CONSTRAINT chk_evaluation_threshold
        CHECK (evaluation_iou_threshold BETWEEN 0 AND 1),

    CONSTRAINT chk_evaluation_counts
        CHECK (
            true_positives >= 0
            AND false_positives >= 0
            AND false_negatives >= 0
        ),

    CONSTRAINT chk_evaluation_metrics
        CHECK (
            (precision_value IS NULL OR
                precision_value BETWEEN 0 AND 1)
            AND
            (recall_value IS NULL OR
                recall_value BETWEEN 0 AND 1)
            AND
            (f1_value IS NULL OR f1_value BETWEEN 0 AND 1)
            AND
            (average_precision IS NULL OR
                average_precision BETWEEN 0 AND 1)
        ),

    CONSTRAINT uq_run_evaluation
        UNIQUE (run_id, evaluation_iou_threshold)
);

-- ------------------------------------------------------------
-- 10. PERFORMANCE INDEXES
-- ------------------------------------------------------------

CREATE INDEX idx_images_dataset
    ON images(dataset_name, image_id);

CREATE INDEX idx_models_status
    ON detection_models(status);

CREATE INDEX idx_runs_image_status
    ON detection_runs(image_id, status);

CREATE INDEX idx_runs_model_created
    ON detection_runs(model_id, created_at DESC);

CREATE INDEX idx_predictions_run_class_confidence
    ON predictions(run_id, class_id, confidence DESC);

CREATE INDEX idx_predictions_retained
    ON predictions(run_id, class_id)
    WHERE status = 'RETAINED';

CREATE INDEX idx_ground_truth_image_class
    ON ground_truth_boxes(image_id, class_id);

CREATE INDEX idx_suppression_run
    ON suppression_events(run_id);

-- ------------------------------------------------------------
-- 11. GEOMETRY FUNCTIONS
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION calculate_iou(
    ax1 DOUBLE PRECISION,
    ay1 DOUBLE PRECISION,
    ax2 DOUBLE PRECISION,
    ay2 DOUBLE PRECISION,
    bx1 DOUBLE PRECISION,
    by1 DOUBLE PRECISION,
    bx2 DOUBLE PRECISION,
    by2 DOUBLE PRECISION
)
RETURNS DOUBLE PRECISION
LANGUAGE plpgsql
IMMUTABLE
STRICT
AS $$
DECLARE
    intersection_width DOUBLE PRECISION;
    intersection_height DOUBLE PRECISION;
    intersection_area DOUBLE PRECISION;
    union_area DOUBLE PRECISION;
BEGIN
    IF ax2 <= ax1 OR ay2 <= ay1
       OR bx2 <= bx1 OR by2 <= by1 THEN
        RAISE EXCEPTION 'Invalid bounding box coordinates';
    END IF;

    IF ax1 IN ('NaN'::FLOAT8, 'Infinity'::FLOAT8,
               '-Infinity'::FLOAT8)
       OR ay1 IN ('NaN'::FLOAT8, 'Infinity'::FLOAT8,
                  '-Infinity'::FLOAT8)
       OR ax2 IN ('NaN'::FLOAT8, 'Infinity'::FLOAT8,
                  '-Infinity'::FLOAT8)
       OR ay2 IN ('NaN'::FLOAT8, 'Infinity'::FLOAT8,
                  '-Infinity'::FLOAT8)
       OR bx1 IN ('NaN'::FLOAT8, 'Infinity'::FLOAT8,
                  '-Infinity'::FLOAT8)
       OR by1 IN ('NaN'::FLOAT8, 'Infinity'::FLOAT8,
                  '-Infinity'::FLOAT8)
       OR bx2 IN ('NaN'::FLOAT8, 'Infinity'::FLOAT8,
                  '-Infinity'::FLOAT8)
       OR by2 IN ('NaN'::FLOAT8, 'Infinity'::FLOAT8,
                  '-Infinity'::FLOAT8) THEN
        RAISE EXCEPTION 'Non-finite bounding box coordinate';
    END IF;

    intersection_width :=
        GREATEST(0, LEAST(ax2, bx2) - GREATEST(ax1, bx1));

    intersection_height :=
        GREATEST(0, LEAST(ay2, by2) - GREATEST(ay1, by1));

    intersection_area :=
        intersection_width * intersection_height;

    union_area :=
        (ax2 - ax1) * (ay2 - ay1)
        + (bx2 - bx1) * (by2 - by1)
        - intersection_area;

    IF union_area <= 0 THEN
        RETURN 0;
    END IF;

    RETURN intersection_area / union_area;
END;
$$;

CREATE OR REPLACE FUNCTION calculate_giou(
    ax1 DOUBLE PRECISION,
    ay1 DOUBLE PRECISION,
    ax2 DOUBLE PRECISION,
    ay2 DOUBLE PRECISION,
    bx1 DOUBLE PRECISION,
    by1 DOUBLE PRECISION,
    bx2 DOUBLE PRECISION,
    by2 DOUBLE PRECISION
)
RETURNS DOUBLE PRECISION
LANGUAGE plpgsql
IMMUTABLE
STRICT
AS $$
DECLARE
    overlap DOUBLE PRECISION;
    enclosing_area DOUBLE PRECISION;
    union_area DOUBLE PRECISION;
BEGIN
    overlap := calculate_iou(
        ax1, ay1, ax2, ay2,
        bx1, by1, bx2, by2
    );

    enclosing_area :=
        (GREATEST(ax2, bx2) - LEAST(ax1, bx1))
        * (GREATEST(ay2, by2) - LEAST(ay1, by1));

    union_area :=
        (ax2 - ax1) * (ay2 - ay1)
        + (bx2 - bx1) * (by2 - by1);

    union_area := union_area -
        (
            GREATEST(0, LEAST(ax2, bx2) - GREATEST(ax1, bx1))
            *
            GREATEST(0, LEAST(ay2, by2) - GREATEST(ay1, by1))
        );

    RETURN overlap -
        (enclosing_area - union_area) / enclosing_area;
END;
$$;

-- ------------------------------------------------------------
-- 12. IMAGE-BOUNDARY VALIDATION TRIGGER
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION validate_box_against_image()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    image_record RECORD;
BEGIN
    SELECT image_width, image_height
    INTO image_record
    FROM images
    WHERE image_id = NEW.image_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Referenced image does not exist';
    END IF;

    IF NEW.x1 < 0 OR NEW.y1 < 0
       OR NEW.x2 > image_record.image_width
       OR NEW.y2 > image_record.image_height THEN
        RAISE EXCEPTION
            'Bounding box exceeds image dimensions: image % x %',
            image_record.image_width,
            image_record.image_height;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_ground_truth_box
BEFORE INSERT OR UPDATE ON ground_truth_boxes
FOR EACH ROW
EXECUTE FUNCTION validate_box_against_image();

CREATE OR REPLACE FUNCTION validate_prediction_box()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    target_image BIGINT;
BEGIN
    SELECT image_id
    INTO target_image
    FROM detection_runs
    WHERE run_id = NEW.run_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Detection run does not exist';
    END IF;

    NEW.created_at := COALESCE(NEW.created_at, NOW());

    IF NOT EXISTS (
        SELECT 1
        FROM images i
        WHERE i.image_id = target_image
          AND NEW.x2 <= i.image_width
          AND NEW.y2 <= i.image_height
    ) THEN
        RAISE EXCEPTION
            'Prediction box exceeds image dimensions';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_prediction_box
BEFORE INSERT OR UPDATE ON predictions
FOR EACH ROW
EXECUTE FUNCTION validate_prediction_box();

-- ------------------------------------------------------------
-- 13. RUN STATE TRANSITION VALIDATION
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION enforce_run_transitions()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.status = 'COMPLETED'
       AND NEW.status <> 'COMPLETED' THEN
        RAISE EXCEPTION
            'Completed runs cannot transition to another state';
    END IF;

    IF OLD.status = 'CANCELLED'
       AND NEW.status <> 'CANCELLED' THEN
        RAISE EXCEPTION
            'Cancelled runs cannot be reopened';
    END IF;

    IF OLD.status = 'FAILED'
       AND NEW.status NOT IN ('FAILED', 'QUEUED') THEN
        RAISE EXCEPTION
            'Failed runs must be explicitly requeued';
    END IF;

    IF OLD.status = 'QUEUED'
       AND NEW.status NOT IN ('QUEUED', 'PROCESSING', 'CANCELLED') THEN
        RAISE EXCEPTION 'Invalid queued run transition';
    END IF;

    IF OLD.status = 'PROCESSING'
       AND NEW.status NOT IN ('PROCESSING', 'COMPLETED',
                              'FAILED', 'CANCELLED') THEN
        RAISE EXCEPTION 'Invalid processing run transition';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_run_state
BEFORE UPDATE OF status ON detection_runs
FOR EACH ROW
EXECUTE FUNCTION enforce_run_transitions();

-- ------------------------------------------------------------
-- 14. CONFIDENCE FILTERING FUNCTION
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION filter_predictions(
    p_run_id BIGINT
)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    threshold_value DOUBLE PRECISION;
    affected INTEGER;
BEGIN
    SELECT confidence_threshold
    INTO threshold_value
    FROM detection_runs
    WHERE run_id = p_run_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Detection run not found';
    END IF;

    UPDATE predictions
    SET status = CASE
        WHEN confidence >= threshold_value
            THEN 'CONFIDENCE_FILTERED'::prediction_status
        ELSE 'SUPPRESSED'::prediction_status
    END
    WHERE run_id = p_run_id
      AND status = 'RAW';

    GET DIAGNOSTICS affected = ROW_COUNT;

    RETURN affected;
END;
$$;

-- ------------------------------------------------------------
-- 15. GREEDY HARD NMS PROCEDURE
--
-- The procedure operates within one run.
-- A higher-confidence box suppresses a same-class box when
-- IoU is strictly greater than the configured threshold.
-- ------------------------------------------------------------

CREATE OR REPLACE PROCEDURE execute_hard_nms(
    p_run_id BIGINT
)
LANGUAGE plpgsql
AS $$
DECLARE
    run_record RECORD;
    selected_record RECORD;
    candidate_record RECORD;
    overlap DOUBLE PRECISION;
    retained_count INTEGER := 0;
BEGIN
    SELECT *
    INTO run_record
    FROM detection_runs
    WHERE run_id = p_run_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Detection run not found';
    END IF;

    IF run_record.status <> 'PROCESSING' THEN
        RAISE EXCEPTION
            'Run must be PROCESSING before NMS';
    END IF;

    IF run_record.suppression <> 'HARD_NMS' THEN
        RAISE EXCEPTION 'Run is not configured for HARD_NMS';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM predictions
        WHERE run_id = p_run_id
          AND status = 'RAW'
    ) THEN
        RAISE EXCEPTION
            'Confidence filtering must run before NMS';
    END IF;

    UPDATE predictions
    SET status = 'CONFIDENCE_FILTERED'
    WHERE run_id = p_run_id
      AND status = 'RETAINED';

    FOR selected_record IN
        SELECT *
        FROM predictions
        WHERE run_id = p_run_id
          AND status = 'CONFIDENCE_FILTERED'
        ORDER BY confidence DESC, prediction_id
    LOOP
        IF selected_record.status <> 'CONFIDENCE_FILTERED' THEN
            CONTINUE;
        END IF;

        IF retained_count >= run_record.max_detections THEN
            EXIT;
        END IF;

        UPDATE predictions
        SET status = 'RETAINED'
        WHERE prediction_id = selected_record.prediction_id;

        retained_count := retained_count + 1;

        FOR candidate_record IN
            SELECT *
            FROM predictions
            WHERE run_id = p_run_id
              AND status = 'CONFIDENCE_FILTERED'
              AND prediction_id <> selected_record.prediction_id
              AND (
                  run_record.class_agnostic
                  OR class_id = selected_record.class_id
              )
            ORDER BY prediction_id
        LOOP
            overlap := calculate_iou(
                selected_record.x1,
                selected_record.y1,
                selected_record.x2,
                selected_record.y2,
                candidate_record.x1,
                candidate_record.y1,
                candidate_record.x2,
                candidate_record.y2
            );

            IF overlap > run_record.iou_threshold THEN
                UPDATE predictions
                SET status = 'SUPPRESSED'
                WHERE prediction_id =
                    candidate_record.prediction_id;

                INSERT INTO suppression_events (
                    run_id,
                    selected_prediction_id,
                    suppressed_prediction_id,
                    overlap_iou,
                    suppression_method
                )
                VALUES (
                    p_run_id,
                    selected_record.prediction_id,
                    candidate_record.prediction_id,
                    overlap,
                    'HARD_NMS'
                );
            END IF;
        END LOOP;
    END LOOP;

    UPDATE predictions
    SET status = 'SUPPRESSED'
    WHERE run_id = p_run_id
      AND status = 'CONFIDENCE_FILTERED';
END;
$$;

-- ------------------------------------------------------------
-- 16. EVALUATION PROCEDURE
--
-- Greedy one-to-one matching:
-- - same image
-- - same class
-- - IoU >= evaluation threshold
-- - one ground-truth box can match only once
-- ------------------------------------------------------------

CREATE OR REPLACE PROCEDURE evaluate_detection_run(
    p_run_id BIGINT,
    p_iou_threshold DOUBLE PRECISION DEFAULT 0.5
)
LANGUAGE plpgsql
AS $$
DECLARE
    run_record RECORD;
    prediction_record RECORD;
    target_record RECORD;
    evaluation_key BIGINT;
    best_target_id BIGINT;
    best_overlap DOUBLE PRECISION;
    current_overlap DOUBLE PRECISION;
    tp INTEGER := 0;
    fp INTEGER := 0;
    fn INTEGER := 0;
    precision_result DOUBLE PRECISION;
    recall_result DOUBLE PRECISION;
    f1_result DOUBLE PRECISION;
BEGIN
    IF p_iou_threshold < 0 OR p_iou_threshold > 1
       OR p_iou_threshold IN (
           'NaN'::FLOAT8,
           'Infinity'::FLOAT8,
           '-Infinity'::FLOAT8
       ) THEN
        RAISE EXCEPTION 'Invalid evaluation IoU threshold';
    END IF;

    SELECT *
    INTO run_record
    FROM detection_runs
    WHERE run_id = p_run_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Detection run not found';
    END IF;

    IF run_record.status <> 'COMPLETED' THEN
        RAISE EXCEPTION
            'Only completed runs can be evaluated';
    END IF;

    INSERT INTO evaluations (
        run_id,
        status,
        evaluation_iou_threshold
    )
    VALUES (
        p_run_id,
        'RUNNING',
        p_iou_threshold
    )
    ON CONFLICT (run_id, evaluation_iou_threshold)
    DO UPDATE SET
        status = 'RUNNING',
        true_positives = 0,
        false_positives = 0,
        false_negatives = 0,
        precision_value = NULL,
        recall_value = NULL,
        f1_value = NULL,
        average_precision = NULL,
        completed_at = NULL
    RETURNING evaluation_id INTO evaluation_key;

    CREATE TEMP TABLE IF NOT EXISTS matched_ground_truth (
        ground_truth_id BIGINT PRIMARY KEY
    ) ON COMMIT DROP;

    TRUNCATE matched_ground_truth;

    FOR prediction_record IN
        SELECT p.*
        FROM predictions p
        WHERE p.run_id = p_run_id
          AND p.status = 'RETAINED'
        ORDER BY p.confidence DESC, p.prediction_id
    LOOP
        best_target_id := NULL;
        best_overlap := p_iou_threshold;

        FOR target_record IN
            SELECT g.*
            FROM ground_truth_boxes g
            WHERE g.image_id = run_record.image_id
              AND g.class_id = prediction_record.class_id
              AND NOT EXISTS (
                  SELECT 1
                  FROM matched_ground_truth m
                  WHERE m.ground_truth_id = g.ground_truth_id
              )
              AND NOT g.difficult
        LOOP
            current_overlap := calculate_iou(
                prediction_record.x1,
                prediction_record.y1,
                prediction_record.x2,
                prediction_record.y2,
                target_record.x1,
                target_record.y1,
                target_record.x2,
                target_record.y2
            );

            IF current_overlap >= best_overlap THEN
                best_overlap := current_overlap;
                best_target_id := target_record.ground_truth_id;
            END IF;
        END LOOP;

        IF best_target_id IS NOT NULL THEN
            INSERT INTO matched_ground_truth(ground_truth_id)
            VALUES (best_target_id);

            tp := tp + 1;
        ELSE
            fp := fp + 1;
        END IF;
    END LOOP;

    SELECT COUNT(*)::INTEGER
    INTO fn
    FROM ground_truth_boxes g
    WHERE g.image_id = run_record.image_id
      AND NOT g.difficult
      AND NOT EXISTS (
          SELECT 1
          FROM matched_ground_truth m
          WHERE m.ground_truth_id = g.ground_truth_id
      );

    precision_result :=
        CASE WHEN tp + fp = 0 THEN 0
             ELSE tp::DOUBLE PRECISION / (tp + fp)
        END;

    recall_result :=
        CASE WHEN tp + fn = 0 THEN 0
             ELSE tp::DOUBLE PRECISION / (tp + fn)
        END;

    f1_result :=
        CASE WHEN precision_result + recall_result = 0 THEN 0
             ELSE 2 * precision_result * recall_result
                  / (precision_result + recall_result)
        END;

    UPDATE evaluations
    SET
        status = 'COMPLETED',
        true_positives = tp,
        false_positives = fp,
        false_negatives = fn,
        precision_value = precision_result,
        recall_value = recall_result,
        f1_value = f1_result,
        completed_at = NOW()
    WHERE evaluation_id = evaluation_key;
END;
$$;

-- ------------------------------------------------------------
-- 17. REPORTING VIEWS
-- ------------------------------------------------------------

CREATE VIEW v_prediction_summary AS
SELECT
    r.run_id,
    i.external_image_key,
    m.model_name,
    m.model_version,
    COUNT(p.prediction_id) AS total_predictions,
    COUNT(*) FILTER (
        WHERE p.status = 'RETAINED'
    ) AS retained_predictions,
    COUNT(*) FILTER (
        WHERE p.status = 'SUPPRESSED'
    ) AS suppressed_predictions,
    AVG(p.confidence) AS average_confidence,
    MAX(p.confidence) AS maximum_confidence
FROM detection_runs r
JOIN images i ON i.image_id = r.image_id
JOIN detection_models m ON m.model_id = r.model_id
LEFT JOIN predictions p ON p.run_id = r.run_id
GROUP BY
    r.run_id,
    i.external_image_key,
    m.model_name,
    m.model_version;

CREATE VIEW v_evaluation_report AS
SELECT
    e.evaluation_id,
    e.run_id,
    m.model_name,
    m.model_version,
    i.external_image_key,
    e.evaluation_iou_threshold,
    e.true_positives,
    e.false_positives,
    e.false_negatives,
    e.precision_value,
    e.recall_value,
    e.f1_value,
    e.status
FROM evaluations e
JOIN detection_runs r ON r.run_id = e.run_id
JOIN detection_models m ON m.model_id = r.model_id
JOIN images i ON i.image_id = r.image_id;

-- ------------------------------------------------------------
-- 18. SAMPLE MODEL AND DATASET
-- ------------------------------------------------------------

INSERT INTO object_classes(class_id, class_name, description)
VALUES
    (0, 'person', 'Human object category'),
    (1, 'bicycle', 'Bicycle object category'),
    (2, 'car', 'Passenger vehicle category'),
    (3, 'bus', 'Bus object category');

INSERT INTO detection_models (
    model_name,
    architecture,
    model_version,
    framework,
    input_width,
    input_height,
    status,
    model_checksum
)
VALUES
(
    'ObjectDetector',
    'YOLO-style detector',
    '1.0.0',
    'PyTorch',
    640,
    640,
    'PRODUCTION',
    'sha256-1234567890abcdef'
);

INSERT INTO images (
    external_image_key,
    storage_uri,
    image_width,
    image_height,
    dataset_name,
    image_checksum
)
VALUES
(
    'image-001',
    'dataset/images/image-001.jpg',
    640,
    480,
    'urban-detection',
    'image-checksum-001'
);

INSERT INTO ground_truth_boxes (
    image_id,
    class_id,
    x1,
    y1,
    x2,
    y2
)
SELECT image_id, 0, 52, 42, 202, 222
FROM images
WHERE external_image_key = 'image-001';

INSERT INTO ground_truth_boxes (
    image_id,
    class_id,
    x1,
    y1,
    x2,
    y2
)
SELECT image_id, 2, 305, 105, 435, 265
FROM images
WHERE external_image_key = 'image-001';

-- ------------------------------------------------------------
-- 19. CREATE DETECTION RUN
-- ------------------------------------------------------------

INSERT INTO detection_runs (
    model_id,
    image_id,
    status,
    confidence_threshold,
    iou_threshold,
    suppression,
    class_agnostic,
    max_detections
)
SELECT
    m.model_id,
    i.image_id,
    'QUEUED',
    0.25,
    0.45,
    'HARD_NMS',
    FALSE,
    100
FROM detection_models m
CROSS JOIN images i
WHERE m.model_name = 'ObjectDetector'
  AND i.external_image_key = 'image-001';

-- ------------------------------------------------------------
-- 20. INSERT RAW DETECTIONS
-- ------------------------------------------------------------

INSERT INTO predictions (
    run_id,
    class_id,
    x1,
    y1,
    x2,
    y2,
    confidence
)
SELECT r.run_id, sample.class_id,
       sample.x1, sample.y1, sample.x2, sample.y2,
       sample.confidence
FROM detection_runs r
CROSS JOIN (
    VALUES
        (0, 50.0, 40.0, 200.0, 220.0, 0.96),
        (0, 55.0, 45.0, 198.0, 218.0, 0.89),
        (2, 300.0, 100.0, 430.0, 260.0, 0.91),
        (0, 500.0, 50.0, 580.0, 130.0, 0.18),
        (0, 610.0, 100.0, 640.0, 220.0, 0.82)
) AS sample(class_id, x1, y1, x2, y2, confidence)
WHERE r.run_id = (
    SELECT MIN(run_id) FROM detection_runs
);

-- ------------------------------------------------------------
-- 21. TRANSACTIONAL WORKFLOW EXECUTION
-- ------------------------------------------------------------

COMMIT;

BEGIN;

UPDATE detection_runs
SET
    status = 'PROCESSING',
    processing_started_at = NOW()
WHERE run_id = 1
  AND status = 'QUEUED';

SELECT filter_predictions(1);

CALL execute_hard_nms(1);

UPDATE detection_runs
SET
    status = 'COMPLETED',
    completed_at = NOW()
WHERE run_id = 1
  AND status = 'PROCESSING';

CALL evaluate_detection_run(1, 0.5);

COMMIT;

-- ------------------------------------------------------------
-- 22. RETRIEVAL
-- ------------------------------------------------------------

SELECT
    p.prediction_id,
    c.class_name,
    p.x1,
    p.y1,
    p.x2,
    p.y2,
    p.confidence,
    p.status
FROM predictions p
JOIN object_classes c ON c.class_id = p.class_id
WHERE p.run_id = 1
ORDER BY p.confidence DESC;

-- ------------------------------------------------------------
-- 23. DIRECT IoU VALIDATION
-- ------------------------------------------------------------

SELECT calculate_iou(
    50, 40, 200, 220,
    55, 45, 198, 218
) AS intersection_over_union;

SELECT calculate_giou(
    50, 40, 200, 220,
    300, 100, 430, 260
) AS generalized_intersection_over_union;

-- ------------------------------------------------------------
-- 24. CONFIDENCE AND NMS AGGREGATION
-- ------------------------------------------------------------

SELECT
    run_id,
    class_id,
    COUNT(*) AS total_boxes,
    COUNT(*) FILTER (
        WHERE status = 'RETAINED'
    ) AS retained_boxes,
    COUNT(*) FILTER (
        WHERE status = 'SUPPRESSED'
    ) AS suppressed_boxes,
    AVG(confidence) AS mean_confidence,
    MAX(confidence) AS highest_confidence
FROM predictions
GROUP BY run_id, class_id
ORDER BY run_id, class_id;

-- ------------------------------------------------------------
-- 25. EVALUATION REPORT
-- ------------------------------------------------------------

SELECT *
FROM v_evaluation_report
ORDER BY evaluation_id;

-- ------------------------------------------------------------
-- 26. SUPPRESSION AUDIT
-- ------------------------------------------------------------

SELECT
    s.event_id,
    s.run_id,
    selected.prediction_id AS selected_id,
    suppressed.prediction_id AS suppressed_id,
    s.overlap_iou,
    s.suppression_method
FROM suppression_events s
JOIN predictions selected
    ON selected.prediction_id = s.selected_prediction_id
JOIN predictions suppressed
    ON suppressed.prediction_id = s.suppressed_prediction_id
ORDER BY s.event_id;

-- ------------------------------------------------------------
-- 27. ELIGIBILITY DETERMINATION
-- ------------------------------------------------------------

SELECT
    r.run_id,
    r.status,
    CASE
        WHEN r.status <> 'COMPLETED'
            THEN 'NOT_ELIGIBLE'
        WHEN r.suppression = 'HARD_NMS'
             AND EXISTS (
                 SELECT 1
                 FROM predictions p
                 WHERE p.run_id = r.run_id
                   AND p.status = 'CONFIDENCE_FILTERED'
             )
            THEN 'NMS_INCOMPLETE'
        WHEN NOT EXISTS (
            SELECT 1
            FROM evaluations e
            WHERE e.run_id = r.run_id
              AND e.status = 'COMPLETED'
        )
            THEN 'EVALUATION_PENDING'
        ELSE 'EVALUATED'
    END AS evaluation_eligibility
FROM detection_runs r
ORDER BY r.run_id;

-- ------------------------------------------------------------
-- 28. FAILURE CASE: INVALID IoU INPUT
-- Run separately when testing expected errors.
-- ------------------------------------------------------------

-- SELECT calculate_iou(10, 10, 5, 20, 0, 0, 10, 10);

-- ------------------------------------------------------------
-- 29. OPERATIONAL METRICS
-- ------------------------------------------------------------

SELECT
    COUNT(*) AS total_runs,
    COUNT(*) FILTER (WHERE status = 'COMPLETED')
        AS completed_runs,
    COUNT(*) FILTER (WHERE status = 'FAILED')
        AS failed_runs,
    COUNT(*) FILTER (WHERE status = 'PROCESSING')
        AS active_runs
FROM detection_runs;

SELECT
    model_id,
    COUNT(*) AS total_runs,
    AVG(
        EXTRACT(EPOCH FROM (completed_at - processing_started_at))
    ) AS average_processing_seconds
FROM detection_runs
WHERE completed_at IS NOT NULL
  AND processing_started_at IS NOT NULL
GROUP BY model_id;

-- ------------------------------------------------------------
-- 30. CLASS-LEVEL GROUND-TRUTH DISTRIBUTION
-- ------------------------------------------------------------

SELECT
    c.class_name,
    COUNT(g.ground_truth_id) AS object_count,
    AVG((g.x2 - g.x1) * (g.y2 - g.y1)) AS average_box_area
FROM object_classes c
LEFT JOIN ground_truth_boxes g
    ON g.class_id = c.class_id
GROUP BY c.class_id, c.class_name
ORDER BY object_count DESC;

-- ------------------------------------------------------------
-- 31. TRANSACTIONAL ROLLBACK DEMONSTRATION
-- ------------------------------------------------------------

BEGIN;

UPDATE predictions
SET status = 'RAW'
WHERE run_id = 1
  AND status = 'SUPPRESSED';

ROLLBACK;

-- The preceding update is deliberately rolled back.
-- Production deployments should use restricted database roles,
-- encrypted storage, access-controlled image URIs and audit logging.
