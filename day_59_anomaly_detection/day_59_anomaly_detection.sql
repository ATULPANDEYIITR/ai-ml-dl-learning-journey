-- Anomaly Detection Relational Governance Model
-- PostgreSQL 15+ / PostgreSQL 18 compatible
--
-- The schema separates:
--   repositories and branches
--   commits and pull requests
--   reviews and comments
--   status checks
--   branch protection
--   detector scores
--   merge eligibility
--
-- The database does not pretend to execute machine-learning algorithms.
-- It persists their observations and enforces repository governance rules
-- at the relational layer.

DROP SCHEMA IF EXISTS anomaly_governance CASCADE;
CREATE SCHEMA anomaly_governance;

SET search_path TO anomaly_governance, public;

CREATE TYPE repository_visibility AS ENUM (
    'private',
    'internal',
    'public'
);

CREATE TYPE pull_request_state AS ENUM (
    'draft',
    'open',
    'closed',
    'merged'
);

CREATE TYPE review_state AS ENUM (
    'commented',
    'approved',
    'changes_requested',
    'dismissed'
);

CREATE TYPE check_state AS ENUM (
    'pending',
    'passed',
    'failed',
    'cancelled'
);

CREATE TYPE merge_strategy AS ENUM (
    'merge_commit',
    'squash',
    'rebase'
);

CREATE TYPE anomaly_detector AS ENUM (
    'isolation_forest',
    'local_outlier_factor',
    'one_class_svm'
);

CREATE TYPE anomaly_state AS ENUM (
    'normal',
    'review',
    'confirmed'
);

CREATE TABLE repositories (
    repository_id BIGSERIAL PRIMARY KEY,
    repository_name TEXT NOT NULL,
    owner_name TEXT NOT NULL,
    visibility repository_visibility NOT NULL DEFAULT 'private',
    default_branch TEXT NOT NULL DEFAULT 'main',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (owner_name, repository_name),
    CHECK (repository_name <> ''),
    CHECK (default_branch <> '')
);

CREATE TABLE users (
    user_id BIGSERIAL PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_admin BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (username <> '')
);

CREATE TABLE repository_members (
    repository_id BIGINT NOT NULL REFERENCES repositories(repository_id)
        ON DELETE CASCADE,
    user_id BIGINT NOT NULL REFERENCES users(user_id)
        ON DELETE CASCADE,
    can_push BOOLEAN NOT NULL DEFAULT FALSE,
    can_merge BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (repository_id, user_id)
);

CREATE TABLE branches (
    branch_id BIGSERIAL PRIMARY KEY,
    repository_id BIGINT NOT NULL REFERENCES repositories(repository_id)
        ON DELETE CASCADE,
    branch_name TEXT NOT NULL,
    is_protected BOOLEAN NOT NULL DEFAULT FALSE,
    head_commit_sha TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (repository_id, branch_name),
    CHECK (branch_name <> ''),
    CHECK (length(head_commit_sha) >= 7)
);

CREATE TABLE commits (
    commit_id BIGSERIAL PRIMARY KEY,
    repository_id BIGINT NOT NULL REFERENCES repositories(repository_id)
        ON DELETE CASCADE,
    commit_sha TEXT NOT NULL,
    author_user_id BIGINT NOT NULL REFERENCES users(user_id),
    parent_commit_sha TEXT,
    changed_files INTEGER NOT NULL,
    additions INTEGER NOT NULL,
    deletions INTEGER NOT NULL,
    committed_at TIMESTAMPTZ NOT NULL,
    UNIQUE (repository_id, commit_sha),
    CHECK (changed_files >= 0),
    CHECK (additions >= 0),
    CHECK (deletions >= 0),
    CHECK (length(commit_sha) >= 7)
);

CREATE TABLE pull_requests (
    pull_request_id BIGSERIAL PRIMARY KEY,
    repository_id BIGINT NOT NULL REFERENCES repositories(repository_id)
        ON DELETE CASCADE,
    source_branch_id BIGINT NOT NULL REFERENCES branches(branch_id),
    target_branch_id BIGINT NOT NULL REFERENCES branches(branch_id),
    author_user_id BIGINT NOT NULL REFERENCES users(user_id),
    title TEXT NOT NULL,
    state pull_request_state NOT NULL DEFAULT 'open',
    is_draft BOOLEAN NOT NULL DEFAULT FALSE,
    source_head_sha TEXT NOT NULL,
    target_base_sha TEXT NOT NULL,
    merge_commit_sha TEXT,
    merge_strategy merge_strategy,
    opened_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    closed_at TIMESTAMPTZ,
    merged_at TIMESTAMPTZ,
    CHECK (source_branch_id <> target_branch_id),
    CHECK (title <> ''),
    CHECK (
        (state = 'merged' AND merged_at IS NOT NULL)
        OR state <> 'merged'
    ),
    CHECK (
        (state = 'closed' AND closed_at IS NOT NULL)
        OR state <> 'closed'
    )
);

CREATE TABLE pull_request_commits (
    pull_request_id BIGINT NOT NULL
        REFERENCES pull_requests(pull_request_id)
        ON DELETE CASCADE,
    commit_id BIGINT NOT NULL
        REFERENCES commits(commit_id)
        ON DELETE RESTRICT,
    sequence_number INTEGER NOT NULL,
    PRIMARY KEY (pull_request_id, commit_id),
    UNIQUE (pull_request_id, sequence_number),
    CHECK (sequence_number > 0)
);

CREATE TABLE review_requests (
    pull_request_id BIGINT NOT NULL
        REFERENCES pull_requests(pull_request_id)
        ON DELETE CASCADE,
    reviewer_user_id BIGINT NOT NULL
        REFERENCES users(user_id),
    requested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    fulfilled_at TIMESTAMPTZ,
    PRIMARY KEY (pull_request_id, reviewer_user_id),
    CHECK (
        fulfilled_at IS NULL
        OR fulfilled_at >= requested_at
    )
);

CREATE TABLE reviews (
    review_id BIGSERIAL PRIMARY KEY,
    pull_request_id BIGINT NOT NULL
        REFERENCES pull_requests(pull_request_id)
        ON DELETE CASCADE,
    reviewer_user_id BIGINT NOT NULL
        REFERENCES users(user_id),
    state review_state NOT NULL,
    commit_sha TEXT NOT NULL,
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    dismissed_at TIMESTAMPTZ,
    dismissal_reason TEXT,
    CHECK (
        state <> 'dismissed'
        OR dismissed_at IS NOT NULL
    )
);

CREATE TABLE review_comments (
    comment_id BIGSERIAL PRIMARY KEY,
    review_id BIGINT NOT NULL
        REFERENCES reviews(review_id)
        ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    line_number INTEGER,
    body TEXT NOT NULL,
    is_resolved BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at TIMESTAMPTZ,
    CHECK (file_path <> ''),
    CHECK (body <> ''),
    CHECK (
        line_number IS NULL
        OR line_number > 0
    ),
    CHECK (
        NOT is_resolved
        OR resolved_at IS NOT NULL
    )
);

CREATE TABLE status_checks (
    status_check_id BIGSERIAL PRIMARY KEY,
    pull_request_id BIGINT NOT NULL
        REFERENCES pull_requests(pull_request_id)
        ON DELETE CASCADE,
    check_name TEXT NOT NULL,
    state check_state NOT NULL,
    commit_sha TEXT NOT NULL,
    completed_at TIMESTAMPTZ,
    UNIQUE (pull_request_id, check_name, commit_sha),
    CHECK (check_name <> ''),
    CHECK (
        state IN ('passed', 'failed', 'cancelled')
        OR completed_at IS NULL
    )
);

CREATE TABLE branch_protection_policies (
    policy_id BIGSERIAL PRIMARY KEY,
    branch_id BIGINT NOT NULL UNIQUE
        REFERENCES branches(branch_id)
        ON DELETE CASCADE,
    required_reviews INTEGER NOT NULL DEFAULT 0,
    dismiss_stale_approvals BOOLEAN NOT NULL DEFAULT TRUE,
    require_conversation_resolution BOOLEAN NOT NULL DEFAULT TRUE,
    require_status_checks BOOLEAN NOT NULL DEFAULT TRUE,
    restrict_direct_push BOOLEAN NOT NULL DEFAULT TRUE,
    restrict_force_push BOOLEAN NOT NULL DEFAULT TRUE,
    restrict_branch_deletion BOOLEAN NOT NULL DEFAULT TRUE,
    require_linear_history BOOLEAN NOT NULL DEFAULT FALSE,
    allow_administrator_bypass BOOLEAN NOT NULL DEFAULT FALSE,
    CHECK (required_reviews >= 0)
);

CREATE TABLE required_status_checks (
    policy_id BIGINT NOT NULL
        REFERENCES branch_protection_policies(policy_id)
        ON DELETE CASCADE,
    check_name TEXT NOT NULL,
    PRIMARY KEY (policy_id, check_name),
    CHECK (check_name <> '')
);

CREATE TABLE approval_requirements (
    requirement_id BIGSERIAL PRIMARY KEY,
    policy_id BIGINT NOT NULL
        REFERENCES branch_protection_policies(policy_id)
        ON DELETE CASCADE,
    minimum_approvals INTEGER NOT NULL,
    minimum_distinct_reviewers INTEGER NOT NULL,
    require_code_owner BOOLEAN NOT NULL DEFAULT FALSE,
    CHECK (minimum_approvals >= 0),
    CHECK (minimum_distinct_reviewers >= 0),
    CHECK (
        minimum_distinct_reviewers <= minimum_approvals
    )
);

CREATE TABLE conversations (
    conversation_id BIGSERIAL PRIMARY KEY,
    pull_request_id BIGINT NOT NULL
        REFERENCES pull_requests(pull_request_id)
        ON DELETE CASCADE,
    opened_by_user_id BIGINT NOT NULL
        REFERENCES users(user_id),
    topic TEXT NOT NULL,
    is_resolved BOOLEAN NOT NULL DEFAULT FALSE,
    opened_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at TIMESTAMPTZ,
    CHECK (topic <> ''),
    CHECK (
        NOT is_resolved
        OR resolved_at IS NOT NULL
    )
);

CREATE TABLE anomaly_observations (
    observation_id BIGSERIAL PRIMARY KEY,
    pull_request_id BIGINT NOT NULL
        REFERENCES pull_requests(pull_request_id)
        ON DELETE CASCADE,
    detector anomaly_detector NOT NULL,
    model_version TEXT NOT NULL,
    feature_schema_version TEXT NOT NULL,
    score NUMERIC(12,8) NOT NULL,
    threshold NUMERIC(12,8) NOT NULL,
    state anomaly_state NOT NULL,
    observed_commit_sha TEXT NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (
        pull_request_id,
        detector,
        model_version,
        observed_commit_sha
    ),
    CHECK (score >= 0),
    CHECK (threshold >= 0)
);

CREATE INDEX idx_pr_repository_state
    ON pull_requests(repository_id, state);

CREATE INDEX idx_reviews_pr_state
    ON reviews(pull_request_id, state);

CREATE INDEX idx_status_checks_pr_state
    ON status_checks(pull_request_id, state);

CREATE INDEX idx_conversations_pr_resolution
    ON conversations(pull_request_id, is_resolved);

CREATE INDEX idx_anomaly_pr_state
    ON anomaly_observations(pull_request_id, state);

CREATE INDEX idx_commits_repository_time
    ON commits(repository_id, committed_at DESC);

-- Sample users.
INSERT INTO users (username, is_admin) VALUES
('atul', FALSE),
('reviewer_one', FALSE),
('reviewer_two', FALSE),
('security_reviewer', FALSE),
('release_admin', TRUE);

INSERT INTO repositories (
    repository_name,
    owner_name,
    visibility
) VALUES (
    'secure-api',
    'engineering',
    'private'
);

INSERT INTO repository_members (
    repository_id,
    user_id,
    can_push,
    can_merge
)
SELECT
    r.repository_id,
    u.user_id,
    CASE WHEN u.username = 'atul' THEN TRUE ELSE FALSE END,
    CASE WHEN u.username IN ('reviewer_one', 'reviewer_two', 'release_admin')
         THEN TRUE ELSE FALSE END
FROM repositories r
CROSS JOIN users u
WHERE r.repository_name = 'secure-api'
  AND u.username IN (
      'atul',
      'reviewer_one',
      'reviewer_two',
      'security_reviewer',
      'release_admin'
  );

INSERT INTO branches (
    repository_id,
    branch_name,
    is_protected,
    head_commit_sha
)
SELECT
    repository_id,
    'main',
    TRUE,
    'aaaaaaaa1111111'
FROM repositories
WHERE repository_name = 'secure-api';

INSERT INTO branches (
    repository_id,
    branch_name,
    is_protected,
    head_commit_sha
)
SELECT
    repository_id,
    'feature/anomaly-monitoring',
    FALSE,
    'bbbbbbbb2222222'
FROM repositories
WHERE repository_name = 'secure-api';

INSERT INTO branch_protection_policies (
    branch_id,
    required_reviews,
    dismiss_stale_approvals,
    require_conversation_resolution,
    require_status_checks,
    restrict_direct_push,
    restrict_force_push,
    restrict_branch_deletion,
    require_linear_history,
    allow_administrator_bypass
)
SELECT
    branch_id,
    2,
    TRUE,
    TRUE,
    TRUE,
    TRUE,
    TRUE,
    TRUE,
    TRUE,
    FALSE
FROM branches
WHERE branch_name = 'main';

INSERT INTO approval_requirements (
    policy_id,
    minimum_approvals,
    minimum_distinct_reviewers,
    require_code_owner
)
SELECT
    policy_id,
    2,
    2,
    TRUE
FROM branch_protection_policies;

INSERT INTO required_status_checks (
    policy_id,
    check_name
)
SELECT
    policy_id,
    check_name
FROM branch_protection_policies
CROSS JOIN (
    VALUES
        ('unit-tests'),
        ('security-scan'),
        ('integration-tests')
) AS checks(check_name);

-- Commits represent actual changesets. A PR references a sequence of commits.
INSERT INTO commits (
    repository_id,
    commit_sha,
    author_user_id,
    parent_commit_sha,
    changed_files,
    additions,
    deletions,
    committed_at
)
SELECT
    r.repository_id,
    'bbbbbbbb2222222',
    u.user_id,
    'aaaaaaaa1111111',
    8,
    420,
    90,
    now() - INTERVAL '4 hours'
FROM repositories r
JOIN users u ON u.username = 'atul'
WHERE r.repository_name = 'secure-api';

INSERT INTO pull_requests (
    repository_id,
    source_branch_id,
    target_branch_id,
    author_user_id,
    title,
    state,
    is_draft,
    source_head_sha,
    target_base_sha
)
SELECT
    r.repository_id,
    source_branch.branch_id,
    target_branch.branch_id,
    u.user_id,
    'Add repository anomaly monitoring',
    'open',
    FALSE,
    source_branch.head_commit_sha,
    target_branch.head_commit_sha
FROM repositories r
JOIN branches source_branch
  ON source_branch.repository_id = r.repository_id
 AND source_branch.branch_name = 'feature/anomaly-monitoring'
JOIN branches target_branch
  ON target_branch.repository_id = r.repository_id
 AND target_branch.branch_name = 'main'
JOIN users u
  ON u.username = 'atul'
WHERE r.repository_name = 'secure-api';

INSERT INTO pull_request_commits (
    pull_request_id,
    commit_id,
    sequence_number
)
SELECT
    pr.pull_request_id,
    c.commit_id,
    1
FROM pull_requests pr
JOIN commits c
  ON c.commit_sha = pr.source_head_sha;

-- Two requested reviewers.
INSERT INTO review_requests (
    pull_request_id,
    reviewer_user_id
)
SELECT
    pr.pull_request_id,
    u.user_id
FROM pull_requests pr
CROSS JOIN users u
WHERE u.username IN ('reviewer_one', 'security_reviewer');

-- Approved review at the current source SHA.
INSERT INTO reviews (
    pull_request_id,
    reviewer_user_id,
    state,
    commit_sha
)
SELECT
    pr.pull_request_id,
    u.user_id,
    'approved',
    pr.source_head_sha
FROM pull_requests pr
JOIN users u
  ON u.username = 'reviewer_one';

-- A second reviewer initially requests changes.
INSERT INTO reviews (
    pull_request_id,
    reviewer_user_id,
    state,
    commit_sha
)
SELECT
    pr.pull_request_id,
    u.user_id,
    'changes_requested',
    pr.source_head_sha
FROM pull_requests pr
JOIN users u
  ON u.username = 'security_reviewer';

INSERT INTO review_comments (
    review_id,
    file_path,
    line_number,
    body,
    is_resolved
)
SELECT
    review_id,
    'src/anomaly_policy.cpp',
    118,
    'Validate the detector threshold before accepting the decision.',
    TRUE
FROM reviews
WHERE state = 'changes_requested';

-- Resolved conversation means the discussion no longer blocks the merge.
INSERT INTO conversations (
    pull_request_id,
    opened_by_user_id,
    topic,
    is_resolved,
    resolved_at
)
SELECT
    pr.pull_request_id,
    u.user_id,
    'Clarify stale approval behavior after a new commit',
    TRUE,
    now()
FROM pull_requests pr
JOIN users u ON u.username = 'security_reviewer';

-- Required checks: all pass for the current source commit.
INSERT INTO status_checks (
    pull_request_id,
    check_name,
    state,
    commit_sha,
    completed_at
)
SELECT
    pr.pull_request_id,
    checks.check_name,
    'passed',
    pr.source_head_sha,
    now()
FROM pull_requests pr
CROSS JOIN (
    VALUES
        ('unit-tests'),
        ('security-scan'),
        ('integration-tests')
) AS checks(check_name);

-- Detector observations demonstrate distinct model evidence.
INSERT INTO anomaly_observations (
    pull_request_id,
    detector,
    model_version,
    feature_schema_version,
    score,
    threshold,
    state,
    observed_commit_sha
)
SELECT
    pr.pull_request_id,
    observation.detector,
    '2026.10',
    'repository-change-v2',
    observation.score,
    observation.threshold,
    observation.state,
    pr.source_head_sha
FROM pull_requests pr
CROSS JOIN (
    VALUES
        ('isolation_forest'::anomaly_detector, 0.31::numeric, 0.47::numeric, 'normal'::anomaly_state),
        ('local_outlier_factor'::anomaly_detector, 1.12::numeric, 1.50::numeric, 'normal'::anomaly_state),
        ('one_class_svm'::anomaly_detector, 0.22::numeric, 0.35::numeric, 'normal'::anomaly_state)
) AS observation(detector, score, threshold, state);

-- Trigger prevents approval by a user who is the PR author.
CREATE OR REPLACE FUNCTION prevent_self_approval()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    pr_author BIGINT;
BEGIN
    SELECT author_user_id
    INTO pr_author
    FROM pull_requests
    WHERE pull_request_id = NEW.pull_request_id;

    IF NEW.reviewer_user_id = pr_author
       AND NEW.state = 'approved' THEN
        RAISE EXCEPTION
            'A pull request author cannot approve their own pull request';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_prevent_self_approval
BEFORE INSERT OR UPDATE ON reviews
FOR EACH ROW
EXECUTE FUNCTION prevent_self_approval();

-- Trigger records that a resolved conversation must have a resolution time.
CREATE OR REPLACE FUNCTION enforce_conversation_resolution()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.is_resolved AND NEW.resolved_at IS NULL THEN
        RAISE EXCEPTION
            'Resolved conversations require resolved_at';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_conversation_resolution
BEFORE INSERT OR UPDATE ON conversations
FOR EACH ROW
EXECUTE FUNCTION enforce_conversation_resolution();

-- View: current approval state for each pull request.
CREATE OR REPLACE VIEW pull_request_approval_status AS
SELECT
    pr.pull_request_id,
    pr.title,
    pr.state,
    pr.source_head_sha,
    COUNT(*) FILTER (
        WHERE r.state = 'approved'
          AND r.dismissed_at IS NULL
          AND r.commit_sha = pr.source_head_sha
    ) AS current_approvals,
    COUNT(DISTINCT r.reviewer_user_id) FILTER (
        WHERE r.state = 'approved'
          AND r.dismissed_at IS NULL
          AND r.commit_sha = pr.source_head_sha
    ) AS distinct_approvers,
    COUNT(*) FILTER (
        WHERE r.state = 'changes_requested'
          AND r.dismissed_at IS NULL
    ) AS active_change_requests
FROM pull_requests pr
LEFT JOIN reviews r
  ON r.pull_request_id = pr.pull_request_id
GROUP BY
    pr.pull_request_id,
    pr.title,
    pr.state,
    pr.source_head_sha;

-- View: status-check state.
CREATE OR REPLACE VIEW pull_request_check_status AS
SELECT
    pr.pull_request_id,
    COUNT(*) FILTER (
        WHERE sc.state = 'passed'
          AND sc.commit_sha = pr.source_head_sha
    ) AS passed_checks,
    COUNT(*) FILTER (
        WHERE sc.state = 'failed'
          AND sc.commit_sha = pr.source_head_sha
    ) AS failed_checks,
    COUNT(*) FILTER (
        WHERE sc.state = 'pending'
          AND sc.commit_sha = pr.source_head_sha
    ) AS pending_checks
FROM pull_requests pr
LEFT JOIN status_checks sc
  ON sc.pull_request_id = pr.pull_request_id
GROUP BY pr.pull_request_id;

-- Function evaluates repository-level merge eligibility.
CREATE OR REPLACE FUNCTION merge_eligibility(
    p_pull_request_id BIGINT
)
RETURNS TABLE (
    eligible BOOLEAN,
    reason TEXT
)
LANGUAGE plpgsql
AS $$
DECLARE
    pr RECORD;
    policy RECORD;
    approvals INTEGER;
    distinct_approvals INTEGER;
    failed_checks INTEGER;
    pending_checks INTEGER;
    unresolved_conversations INTEGER;
    active_change_requests INTEGER;
    required_checks INTEGER;
    passing_required_checks INTEGER;
BEGIN
    SELECT
        pr0.*,
        target.branch_id AS target_branch_id
    INTO pr
    FROM pull_requests pr0
    JOIN branches target
      ON target.branch_id = pr0.target_branch_id
    WHERE pr0.pull_request_id = p_pull_request_id;

    IF NOT FOUND THEN
        RETURN QUERY
        SELECT FALSE, 'Pull request does not exist';
        RETURN;
    END IF;

    IF pr.state <> 'open' OR pr.is_draft THEN
        RETURN QUERY
        SELECT FALSE, 'Pull request is not mergeable in its current state';
        RETURN;
    END IF;

    SELECT bp.*
    INTO policy
    FROM branch_protection_policies bp
    WHERE bp.branch_id = pr.target_branch_id;

    IF NOT FOUND THEN
        RETURN QUERY
        SELECT TRUE, 'Target branch has no protection policy';
        RETURN;
    END IF;

    SELECT
        COUNT(*) FILTER (
            WHERE r.state = 'approved'
              AND r.dismissed_at IS NULL
              AND r.commit_sha = pr.source_head_sha
        ),
        COUNT(DISTINCT r.reviewer_user_id) FILTER (
            WHERE r.state = 'approved'
              AND r.dismissed_at IS NULL
              AND r.commit_sha = pr.source_head_sha
        ),
        COUNT(*) FILTER (
            WHERE r.state = 'changes_requested'
              AND r.dismissed_at IS NULL
        )
    INTO approvals, distinct_approvals, active_change_requests
    FROM reviews r
    WHERE r.pull_request_id = p_pull_request_id;

    IF active_change_requests > 0 THEN
        RETURN QUERY
        SELECT FALSE, 'Active change requests remain';
        RETURN;
    END IF;

    SELECT COUNT(*)
    INTO unresolved_conversations
    FROM conversations
    WHERE pull_request_id = p_pull_request_id
      AND NOT is_resolved;

    IF policy.require_conversation_resolution
       AND unresolved_conversations > 0 THEN
        RETURN QUERY
        SELECT FALSE, 'Unresolved review conversations remain';
        RETURN;
    END IF;

    IF approvals < policy.required_reviews THEN
        RETURN QUERY
        SELECT FALSE, 'Required approvals are not satisfied';
        RETURN;
    END IF;

    SELECT COUNT(*)
    INTO required_checks
    FROM required_status_checks rsc
    WHERE rsc.policy_id = policy.policy_id;

    SELECT COUNT(*)
    INTO passing_required_checks
    FROM required_status_checks rsc
    JOIN status_checks sc
      ON sc.pull_request_id = p_pull_request_id
     AND sc.check_name = rsc.check_name
     AND sc.commit_sha = pr.source_head_sha
     AND sc.state = 'passed'
    WHERE rsc.policy_id = policy.policy_id;

    SELECT COUNT(*)
    INTO failed_checks
    FROM status_checks sc
    WHERE sc.pull_request_id = p_pull_request_id
      AND sc.commit_sha = pr.source_head_sha
      AND sc.state = 'failed';

    SELECT COUNT(*)
    INTO pending_checks
    FROM status_checks sc
    WHERE sc.pull_request_id = p_pull_request_id
      AND sc.commit_sha = pr.source_head_sha
      AND sc.state = 'pending';

    IF policy.require_status_checks
       AND (
           passing_required_checks < required_checks
           OR failed_checks > 0
           OR pending_checks > 0
       ) THEN
        RETURN QUERY
        SELECT FALSE, 'Required status checks are not satisfied';
        RETURN;
    END IF;

    RETURN QUERY
    SELECT TRUE, 'Pull request satisfies configured merge conditions';
END;
$$;

-- CTE-based governance report.
WITH latest_anomalies AS (
    SELECT DISTINCT ON (
        pull_request_id,
        detector
    )
        pull_request_id,
        detector,
        score,
        threshold,
        state,
        observed_at
    FROM anomaly_observations
    ORDER BY
        pull_request_id,
        detector,
        observed_at DESC
),
anomaly_summary AS (
    SELECT
        pull_request_id,
        COUNT(*) FILTER (
            WHERE state IN ('review', 'confirmed')
        ) AS detectors_flagging,
        MAX(score) AS highest_score
    FROM latest_anomalies
    GROUP BY pull_request_id
)
SELECT
    pr.pull_request_id,
    pr.title,
    COALESCE(a.detectors_flagging, 0) AS detectors_flagging,
    COALESCE(a.highest_score, 0) AS highest_anomaly_score,
    eligibility.eligible,
    eligibility.reason
FROM pull_requests pr
LEFT JOIN anomaly_summary a
  ON a.pull_request_id = pr.pull_request_id
CROSS JOIN LATERAL merge_eligibility(
    pr.pull_request_id
) AS eligibility
ORDER BY
    pr.pull_request_id;

-- Retrieval: review decisions and current commit SHA.
SELECT
    pr.pull_request_id,
    pr.title,
    u.username AS reviewer,
    r.state,
    r.commit_sha,
    r.submitted_at
FROM reviews r
JOIN pull_requests pr
  ON pr.pull_request_id = r.pull_request_id
JOIN users u
  ON u.user_id = r.reviewer_user_id
ORDER BY
    pr.pull_request_id,
    r.submitted_at;

-- Validation: identify stale approvals.
SELECT
    pr.pull_request_id,
    u.username AS reviewer,
    r.commit_sha AS approved_commit,
    pr.source_head_sha AS current_commit
FROM reviews r
JOIN pull_requests pr
  ON pr.pull_request_id = r.pull_request_id
JOIN users u
  ON u.user_id = r.reviewer_user_id
WHERE r.state = 'approved'
  AND r.dismissed_at IS NULL
  AND r.commit_sha <> pr.source_head_sha;

-- Aggregation: detector behavior.
SELECT
    detector,
    COUNT(*) AS observations,
    AVG(score) AS average_score,
    MAX(score) AS maximum_score,
    COUNT(*) FILTER (
        WHERE state IN ('review', 'confirmed')
    ) AS flagged_observations
FROM anomaly_observations
GROUP BY detector
ORDER BY maximum_score DESC;

-- Workflow report: unresolved conversations.
SELECT
    pr.pull_request_id,
    pr.title,
    COUNT(*) AS unresolved_conversations
FROM conversations c
JOIN pull_requests pr
  ON pr.pull_request_id = c.pull_request_id
WHERE NOT c.is_resolved
GROUP BY
    pr.pull_request_id,
    pr.title
HAVING COUNT(*) > 0;

-- Explicit transactional example. The update is rolled back because it
-- represents a failed governance attempt rather than a legitimate merge.
BEGIN;

SELECT *
FROM merge_eligibility(1);

-- A merge operation would update the PR only after eligibility is true.
-- This transaction intentionally performs no unauthorized state change.

ROLLBACK;

-- Final eligibility query.
SELECT
    pr.pull_request_id,
    pr.title,
    result.eligible,
    result.reason
FROM pull_requests pr
CROSS JOIN LATERAL merge_eligibility(
    pr.pull_request_id
) AS result;
