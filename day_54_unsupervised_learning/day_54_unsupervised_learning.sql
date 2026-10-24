DROP SCHEMA IF EXISTS unsupervised_learning CASCADE;
CREATE SCHEMA unsupervised_learning;
SET search_path = unsupervised_learning, public;

CREATE TYPE observation_status AS ENUM ('ACTIVE', 'REJECTED', 'ANOMALY');
CREATE TYPE model_status AS ENUM ('DRAFT', 'TRAINED', 'FAILED', 'RETIRED');
CREATE TYPE review_state AS ENUM (
    'PENDING',
    'APPROVED',
    'CHANGES_REQUESTED',
    'COMMENTED',
    'DISMISSED'
);
CREATE TYPE check_state AS ENUM ('PASS', 'FAIL', 'PENDING');
CREATE TYPE merge_state AS ENUM (
    'OPEN',
    'BLOCKED',
    'MERGEABLE',
    'MERGED',
    'CLOSED'
);

CREATE TABLE datasets (
    dataset_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_name TEXT NOT NULL UNIQUE,
    feature_count INTEGER NOT NULL CHECK (feature_count > 0),
    row_count INTEGER NOT NULL CHECK (row_count >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE observations (
    observation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    external_key TEXT NOT NULL,
    features DOUBLE PRECISION[] NOT NULL,
    status observation_status NOT NULL DEFAULT 'ACTIVE',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (dataset_id, external_key),
    CHECK (cardinality(features) > 0)
);

CREATE INDEX idx_observations_dataset_status
    ON observations(dataset_id, status);

CREATE TABLE models (
    model_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL REFERENCES datasets(dataset_id),
    model_type TEXT NOT NULL CHECK (
        model_type IN ('K_MEANS', 'PCA', 'DISTANCE_ANOMALY')
    ),
    status model_status NOT NULL DEFAULT 'DRAFT',
    parameter_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    trained_at TIMESTAMPTZ
);

CREATE TABLE clusters (
    cluster_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_id BIGINT NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
    cluster_number INTEGER NOT NULL CHECK (cluster_number >= 0),
    centroid DOUBLE PRECISION[] NOT NULL,
    UNIQUE (model_id, cluster_number)
);

CREATE TABLE cluster_memberships (
    model_id BIGINT NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
    observation_id BIGINT NOT NULL REFERENCES observations(observation_id) ON DELETE CASCADE,
    cluster_id BIGINT NOT NULL REFERENCES clusters(cluster_id) ON DELETE CASCADE,
    distance_to_centroid DOUBLE PRECISION NOT NULL CHECK (distance_to_centroid >= 0),
    PRIMARY KEY (model_id, observation_id)
);

CREATE INDEX idx_cluster_memberships_cluster
    ON cluster_memberships(cluster_id);

CREATE TABLE anomaly_scores (
    model_id BIGINT NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
    observation_id BIGINT NOT NULL REFERENCES observations(observation_id) ON DELETE CASCADE,
    score DOUBLE PRECISION NOT NULL CHECK (score >= 0),
    threshold DOUBLE PRECISION NOT NULL CHECK (threshold >= 0),
    is_anomaly BOOLEAN NOT NULL,
    PRIMARY KEY (model_id, observation_id)
);

CREATE TABLE repositories (
    repository_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    repository_name TEXT NOT NULL UNIQUE
);

CREATE TABLE branches (
    branch_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    repository_id BIGINT NOT NULL REFERENCES repositories(repository_id) ON DELETE CASCADE,
    branch_name TEXT NOT NULL,
    protected BOOLEAN NOT NULL DEFAULT false,
    UNIQUE (repository_id, branch_name)
);

CREATE TABLE commits (
    commit_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    repository_id BIGINT NOT NULL REFERENCES repositories(repository_id) ON DELETE CASCADE,
    commit_hash TEXT NOT NULL,
    parent_hash TEXT,
    UNIQUE (repository_id, commit_hash)
);

CREATE TABLE pull_requests (
    pull_request_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    repository_id BIGINT NOT NULL REFERENCES repositories(repository_id),
    source_branch_id BIGINT NOT NULL REFERENCES branches(branch_id),
    target_branch_id BIGINT NOT NULL REFERENCES branches(branch_id),
    title TEXT NOT NULL,
    draft BOOLEAN NOT NULL DEFAULT false,
    has_conflicts BOOLEAN NOT NULL DEFAULT false,
    conversations_resolved BOOLEAN NOT NULL DEFAULT false,
    merge_state merge_state NOT NULL DEFAULT 'OPEN',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (source_branch_id <> target_branch_id)
);

CREATE INDEX idx_pull_requests_repository_state
    ON pull_requests(repository_id, merge_state);

CREATE TABLE reviewers (
    reviewer_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    reviewer_name TEXT NOT NULL UNIQUE,
    active BOOLEAN NOT NULL DEFAULT true
);

CREATE TABLE reviews (
    review_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pull_request_id BIGINT NOT NULL REFERENCES pull_requests(pull_request_id) ON DELETE CASCADE,
    reviewer_id BIGINT NOT NULL REFERENCES reviewers(reviewer_id),
    state review_state NOT NULL,
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_reviews_pr_state
    ON reviews(pull_request_id, state);

CREATE TABLE review_comments (
    comment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    review_id BIGINT NOT NULL REFERENCES reviews(review_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    line_number INTEGER CHECK (line_number > 0),
    comment_text TEXT NOT NULL,
    resolved BOOLEAN NOT NULL DEFAULT false
);

CREATE TABLE status_checks (
    check_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pull_request_id BIGINT NOT NULL REFERENCES pull_requests(pull_request_id) ON DELETE CASCADE,
    check_name TEXT NOT NULL,
    state check_state NOT NULL,
    UNIQUE (pull_request_id, check_name)
);

CREATE TABLE branch_protection_policies (
    policy_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    branch_id BIGINT NOT NULL UNIQUE REFERENCES branches(branch_id) ON DELETE CASCADE,
    required_approvals INTEGER NOT NULL DEFAULT 0 CHECK (required_approvals >= 0),
    require_all_checks BOOLEAN NOT NULL DEFAULT true,
    require_resolved_conversations BOOLEAN NOT NULL DEFAULT true,
    allow_direct_push BOOLEAN NOT NULL DEFAULT false,
    allow_force_push BOOLEAN NOT NULL DEFAULT false,
    require_linear_history BOOLEAN NOT NULL DEFAULT false
);

CREATE OR REPLACE FUNCTION validate_feature_dimensions()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    expected_count INTEGER;
BEGIN
    SELECT feature_count
    INTO expected_count
    FROM datasets
    WHERE dataset_id = NEW.dataset_id;

    IF expected_count IS NULL THEN
        RAISE EXCEPTION 'Dataset % does not exist', NEW.dataset_id;
    END IF;

    IF cardinality(NEW.features) <> expected_count THEN
        RAISE EXCEPTION
            'Observation has % features but dataset requires %',
            cardinality(NEW.features),
            expected_count;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_feature_dimensions
BEFORE INSERT OR UPDATE ON observations
FOR EACH ROW
EXECUTE FUNCTION validate_feature_dimensions();

CREATE OR REPLACE FUNCTION refresh_pull_request_state(p_pr_id BIGINT)
RETURNS merge_state
LANGUAGE plpgsql
AS $$
DECLARE
    target_branch BIGINT;
    required_approvals INTEGER;
    approval_count INTEGER;
    failed_checks INTEGER;
    unresolved_comments INTEGER;
    draft_flag BOOLEAN;
    conflict_flag BOOLEAN;
    conversations_flag BOOLEAN;
    next_state merge_state;
BEGIN
    SELECT target_branch_id,
           draft,
           has_conflicts,
           conversations_resolved
    INTO target_branch, draft_flag, conflict_flag, conversations_flag
    FROM pull_requests
    WHERE pull_request_id = p_pr_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Pull Request % does not exist', p_pr_id;
    END IF;

    SELECT required_approvals
    INTO required_approvals
    FROM branch_protection_policies
    WHERE branch_id = target_branch;

    required_approvals := COALESCE(required_approvals, 0);

    SELECT count(*)
    INTO approval_count
    FROM reviews r
    JOIN reviewers rv ON rv.reviewer_id = r.reviewer_id
    WHERE r.pull_request_id = p_pr_id
      AND rv.active
      AND r.state = 'APPROVED';

    SELECT count(*)
    INTO failed_checks
    FROM status_checks
    WHERE pull_request_id = p_pr_id
      AND state <> 'PASS';

    SELECT count(*)
    INTO unresolved_comments
    FROM review_comments rc
    JOIN reviews r ON r.review_id = rc.review_id
    WHERE r.pull_request_id = p_pr_id
      AND NOT rc.resolved;

    IF draft_flag
       OR conflict_flag
       OR approval_count < required_approvals
       OR failed_checks > 0
       OR unresolved_comments > 0
       OR NOT conversations_flag
    THEN
        next_state := 'BLOCKED';
    ELSE
        next_state := 'MERGEABLE';
    END IF;

    UPDATE pull_requests
    SET merge_state = next_state
    WHERE pull_request_id = p_pr_id;

    RETURN next_state;
END;
$$;

INSERT INTO datasets(dataset_name, feature_count, row_count)
VALUES ('customer_behavior', 3, 10);

INSERT INTO observations(dataset_id, external_key, features)
SELECT dataset_id, x.external_key, x.features
FROM datasets d
CROSS JOIN (
    VALUES
        ('customer-01', ARRAY[1.0, 1.2, 0.9]::double precision[]),
        ('customer-02', ARRAY[1.1, 0.8, 1.0]::double precision[]),
        ('customer-03', ARRAY[0.9, 1.0, 1.2]::double precision[]),
        ('customer-04', ARRAY[7.8, 8.1, 7.5]::double precision[]),
        ('customer-05', ARRAY[8.2, 7.9, 8.4]::double precision[]),
        ('customer-06', ARRAY[7.5, 8.3, 8.0]::double precision[]),
        ('customer-07', ARRAY[4.2, 4.0, 4.1]::double precision[]),
        ('customer-08', ARRAY[4.5, 4.3, 4.2]::double precision[]),
        ('customer-09', ARRAY[4.0, 4.4, 4.5]::double precision[]),
        ('customer-10', ARRAY[25.0, 1.0, 0.5]::double precision[])
) AS x(external_key, features)
WHERE d.dataset_name = 'customer_behavior';

INSERT INTO models(dataset_id, model_type, status, parameter_json, trained_at)
SELECT dataset_id, 'K_MEANS', 'TRAINED',
       '{"k": 3, "distance": "euclidean"}'::jsonb,
       now()
FROM datasets
WHERE dataset_name = 'customer_behavior';

INSERT INTO repositories(repository_name)
VALUES ('unsupervised-learning-platform');

INSERT INTO branches(repository_id, branch_name, protected)
SELECT repository_id, 'main', true
FROM repositories
WHERE repository_name = 'unsupervised-learning-platform';

INSERT INTO branches(repository_id, branch_name, protected)
SELECT repository_id, 'feature/clustering', false
FROM repositories
WHERE repository_name = 'unsupervised-learning-platform';

INSERT INTO branch_protection_policies(
    branch_id,
    required_approvals,
    require_all_checks,
    require_resolved_conversations,
    allow_direct_push,
    allow_force_push,
    require_linear_history
)
SELECT branch_id, 2, true, true, false, false, true
FROM branches
WHERE branch_name = 'main';

INSERT INTO reviewers(reviewer_name)
VALUES ('reviewer-a'), ('reviewer-b'), ('reviewer-c');

INSERT INTO pull_requests(
    repository_id,
    source_branch_id,
    target_branch_id,
    title,
    draft,
    has_conflicts,
    conversations_resolved
)
SELECT
    r.repository_id,
    source.branch_id,
    target.branch_id,
    'Add customer clustering pipeline',
    false,
    false,
    true
FROM repositories r
JOIN branches source
  ON source.repository_id = r.repository_id
 AND source.branch_name = 'feature/clustering'
JOIN branches target
  ON target.repository_id = r.repository_id
 AND target.branch_name = 'main'
WHERE r.repository_name = 'unsupervised-learning-platform';

INSERT INTO reviews(pull_request_id, reviewer_id, state)
SELECT pr.pull_request_id, rv.reviewer_id, 'APPROVED'
FROM pull_requests pr
CROSS JOIN reviewers rv
WHERE pr.title = 'Add customer clustering pipeline'
  AND rv.reviewer_name IN ('reviewer-a', 'reviewer-b');

INSERT INTO reviews(pull_request_id, reviewer_id, state)
SELECT pr.pull_request_id, rv.reviewer_id, 'COMMENTED'
FROM pull_requests pr
JOIN reviewers rv ON rv.reviewer_name = 'reviewer-c'
WHERE pr.title = 'Add customer clustering pipeline';

INSERT INTO status_checks(pull_request_id, check_name, state)
SELECT pr.pull_request_id, checks.name, 'PASS'
FROM pull_requests pr
CROSS JOIN (
    VALUES ('unit-tests'), ('security-scan'), ('data-quality')
) AS checks(name)
WHERE pr.title = 'Add customer clustering pipeline';

SELECT refresh_pull_request_state(pull_request_id)
FROM pull_requests
WHERE title = 'Add customer clustering pipeline';

CREATE VIEW merge_eligibility_report AS
SELECT
    pr.pull_request_id,
    pr.title,
    pr.merge_state,
    bp.required_approvals,
    (
        SELECT count(*)
        FROM reviews r
        JOIN reviewers rv ON rv.reviewer_id = r.reviewer_id
        WHERE r.pull_request_id = pr.pull_request_id
          AND rv.active
          AND r.state = 'APPROVED'
    ) AS eligible_approvals,
    (
        SELECT count(*)
        FROM status_checks sc
        WHERE sc.pull_request_id = pr.pull_request_id
          AND sc.state <> 'PASS'
    ) AS failing_or_pending_checks,
    (
        SELECT count(*)
        FROM review_comments rc
        JOIN reviews r ON r.review_id = rc.review_id
        WHERE r.pull_request_id = pr.pull_request_id
          AND NOT rc.resolved
    ) AS unresolved_comments
FROM pull_requests pr
JOIN branch_protection_policies bp
  ON bp.branch_id = pr.target_branch_id;

-- Retrieval: inspect observations belonging to the active dataset.
SELECT
    o.external_key,
    o.features,
    o.status
FROM observations o
JOIN datasets d ON d.dataset_id = o.dataset_id
WHERE d.dataset_name = 'customer_behavior'
ORDER BY o.external_key;

-- Aggregation: count observations by operational state.
SELECT status, count(*) AS observation_count
FROM observations
GROUP BY status
ORDER BY status;

-- Workflow evaluation: show why a Pull Request is currently mergeable or blocked.
SELECT *
FROM merge_eligibility_report
ORDER BY pull_request_id;

-- Validation example: duplicate-safe reviewer approval counting.
SELECT
    pr.title,
    count(*) FILTER (WHERE r.state = 'APPROVED') AS approvals,
    bp.required_approvals,
    count(*) FILTER (WHERE r.state = 'APPROVED') >= bp.required_approvals
        AS approval_requirement_satisfied
FROM pull_requests pr
JOIN reviews r ON r.pull_request_id = pr.pull_request_id
JOIN branch_protection_policies bp
  ON bp.branch_id = pr.target_branch_id
GROUP BY pr.title, bp.required_approvals;

-- Transactional example: a review decision and conversation resolution
-- become visible together or not at all.
BEGIN;

UPDATE review_comments
SET resolved = true
WHERE comment_id IN (
    SELECT rc.comment_id
    FROM review_comments rc
    JOIN reviews r ON r.review_id = rc.review_id
    JOIN pull_requests pr ON pr.pull_request_id = r.pull_request_id
    WHERE pr.title = 'Add customer clustering pipeline'
);

UPDATE pull_requests
SET conversations_resolved = true
WHERE title = 'Add customer clustering pipeline';

SELECT refresh_pull_request_state(pull_request_id)
FROM pull_requests
WHERE title = 'Add customer clustering pipeline';

COMMIT;
