-- PostgreSQL 15+
-- Logistic Regression Governance Data Model
--
-- The relational model separates:
-- Pull Requests from reviews,
-- reviews from approval eligibility,
-- approval decisions from review comments,
-- branch protection from Pull Request state,
-- status checks from merge eligibility,
-- and model risk scores from repository policy.
--
-- This script is intentionally executable as a complete PostgreSQL example.

DROP SCHEMA IF EXISTS logistic_regression_governance CASCADE;

CREATE SCHEMA logistic_regression_governance;

SET search_path TO logistic_regression_governance;

CREATE TABLE repositories (
    repository_id BIGSERIAL PRIMARY KEY,
    repository_name TEXT NOT NULL UNIQUE,
    default_branch TEXT NOT NULL DEFAULT 'main',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (length(trim(repository_name)) > 0),
    CHECK (length(trim(default_branch)) > 0)
);

CREATE TABLE branches (
    branch_id BIGSERIAL PRIMARY KEY,
    repository_id BIGINT NOT NULL
        REFERENCES repositories(repository_id)
        ON DELETE CASCADE,
    branch_name TEXT NOT NULL,
    is_protected BOOLEAN NOT NULL DEFAULT FALSE,
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (repository_id, branch_name),
    CHECK (length(trim(branch_name)) > 0)
);

CREATE TABLE developers (
    developer_id BIGSERIAL PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    CHECK (length(trim(username)) > 0),
    CHECK (length(trim(display_name)) > 0)
);

CREATE TABLE pull_requests (
    pull_request_id BIGSERIAL PRIMARY KEY,
    repository_id BIGINT NOT NULL
        REFERENCES repositories(repository_id)
        ON DELETE CASCADE,
    number INTEGER NOT NULL,
    source_branch_id BIGINT NOT NULL
        REFERENCES branches(branch_id),
    target_branch_id BIGINT NOT NULL
        REFERENCES branches(branch_id),
    author_id BIGINT NOT NULL
        REFERENCES developers(developer_id),
    title TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'draft',
    is_draft BOOLEAN NOT NULL DEFAULT TRUE,
    changed_files INTEGER NOT NULL DEFAULT 0,
    additions INTEGER NOT NULL DEFAULT 0,
    deletions INTEGER NOT NULL DEFAULT 0,
    commit_count INTEGER NOT NULL DEFAULT 0,
    branch_synchronized BOOLEAN NOT NULL DEFAULT FALSE,
    merge_conflict BOOLEAN NOT NULL DEFAULT FALSE,
    conversations_resolved BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (repository_id, number),
    CHECK (number > 0),
    CHECK (state IN ('draft', 'open', 'closed', 'merged')),
    CHECK (changed_files >= 0),
    CHECK (additions >= 0),
    CHECK (deletions >= 0),
    CHECK (commit_count >= 0),
    CHECK (source_branch_id <> target_branch_id),
    CHECK (
        (state = 'draft' AND is_draft = TRUE)
        OR
        (state <> 'draft' AND is_draft = FALSE)
    )
);

CREATE TABLE commits (
    commit_id BIGSERIAL PRIMARY KEY,
    pull_request_id BIGINT NOT NULL
        REFERENCES pull_requests(pull_request_id)
        ON DELETE CASCADE,
    commit_hash TEXT NOT NULL,
    author_id BIGINT NOT NULL
        REFERENCES developers(developer_id),
    message TEXT NOT NULL,
    committed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (commit_hash),
    CHECK (length(commit_hash) >= 7),
    CHECK (length(trim(message)) > 0)
);

CREATE TABLE reviews (
    review_id BIGSERIAL PRIMARY KEY,
    pull_request_id BIGINT NOT NULL
        REFERENCES pull_requests(pull_request_id)
        ON DELETE CASCADE,
    reviewer_id BIGINT NOT NULL
        REFERENCES developers(developer_id),
    review_state TEXT NOT NULL,
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    dismissed_at TIMESTAMPTZ,
    dismissal_reason TEXT,
    UNIQUE (pull_request_id, reviewer_id, submitted_at),
    CHECK (
        review_state IN (
            'commented',
            'changes_requested',
            'approved',
            'dismissed'
        )
    ),
    CHECK (
        review_state <> 'dismissed'
        OR dismissed_at IS NOT NULL
    )
);

CREATE TABLE review_comments (
    comment_id BIGSERIAL PRIMARY KEY,
    review_id BIGINT NOT NULL
        REFERENCES reviews(review_id)
        ON DELETE CASCADE,
    author_id BIGINT NOT NULL
        REFERENCES developers(developer_id),
    file_path TEXT,
    line_number INTEGER,
    body TEXT NOT NULL,
    is_resolved BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (length(trim(body)) > 0),
    CHECK (
        line_number IS NULL
        OR line_number > 0
    )
);

CREATE TABLE status_checks (
    status_check_id BIGSERIAL PRIMARY KEY,
    pull_request_id BIGINT NOT NULL
        REFERENCES pull_requests(pull_request_id)
        ON DELETE CASCADE,
    check_name TEXT NOT NULL,
    status TEXT NOT NULL,
    required BOOLEAN NOT NULL DEFAULT FALSE,
    completed_at TIMESTAMPTZ,
    UNIQUE (pull_request_id, check_name),
    CHECK (
        status IN ('pending', 'passed', 'failed', 'cancelled')
    )
);

CREATE TABLE branch_protection_policies (
    policy_id BIGSERIAL PRIMARY KEY,
    branch_id BIGINT NOT NULL UNIQUE
        REFERENCES branches(branch_id)
        ON DELETE CASCADE,
    required_approvals INTEGER NOT NULL DEFAULT 0,
    require_status_checks BOOLEAN NOT NULL DEFAULT TRUE,
    require_conversation_resolution BOOLEAN NOT NULL DEFAULT TRUE,
    restrict_direct_push BOOLEAN NOT NULL DEFAULT TRUE,
    restrict_force_push BOOLEAN NOT NULL DEFAULT TRUE,
    restrict_branch_deletion BOOLEAN NOT NULL DEFAULT TRUE,
    require_linear_history BOOLEAN NOT NULL DEFAULT FALSE,
    allow_administrator_bypass BOOLEAN NOT NULL DEFAULT FALSE,
    CHECK (required_approvals >= 0)
);

CREATE TABLE approval_policies (
    approval_policy_id BIGSERIAL PRIMARY KEY,
    repository_id BIGINT NOT NULL
        REFERENCES repositories(repository_id)
        ON DELETE CASCADE,
    policy_name TEXT NOT NULL,
    minimum_approvals INTEGER NOT NULL,
    require_unique_reviewers BOOLEAN NOT NULL DEFAULT TRUE,
    dismiss_stale_approvals BOOLEAN NOT NULL DEFAULT TRUE,
    UNIQUE (repository_id, policy_name),
    CHECK (minimum_approvals >= 0)
);

CREATE TABLE reviewer_eligibility (
    repository_id BIGINT NOT NULL
        REFERENCES repositories(repository_id)
        ON DELETE CASCADE,
    developer_id BIGINT NOT NULL
        REFERENCES developers(developer_id)
        ON DELETE CASCADE,
    can_approve BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (repository_id, developer_id)
);

CREATE TABLE logistic_risk_scores (
    risk_score_id BIGSERIAL PRIMARY KEY,
    pull_request_id BIGINT NOT NULL UNIQUE
        REFERENCES pull_requests(pull_request_id)
        ON DELETE CASCADE,
    model_version TEXT NOT NULL,
    logit_value DOUBLE PRECISION NOT NULL,
    probability DOUBLE PRECISION NOT NULL,
    classification_threshold DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (probability >= 0 AND probability <= 1),
    CHECK (
        classification_threshold > 0
        AND classification_threshold < 1
    )
);

CREATE TABLE merge_attempts (
    merge_attempt_id BIGSERIAL PRIMARY KEY,
    pull_request_id BIGINT NOT NULL
        REFERENCES pull_requests(pull_request_id)
        ON DELETE CASCADE,
    attempted_by BIGINT NOT NULL
        REFERENCES developers(developer_id),
    eligible BOOLEAN NOT NULL,
    blocked_reason TEXT,
    attempted_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_pull_requests_target_state
    ON pull_requests(target_branch_id, state);

CREATE INDEX idx_reviews_pull_request_state
    ON reviews(pull_request_id, review_state);

CREATE INDEX idx_review_comments_unresolved
    ON review_comments(review_id)
    WHERE is_resolved = FALSE;

CREATE INDEX idx_status_checks_required_status
    ON status_checks(pull_request_id, required, status);

CREATE INDEX idx_risk_scores_probability
    ON logistic_risk_scores(probability);

CREATE VIEW active_eligible_approvals AS
SELECT
    r.pull_request_id,
    r.reviewer_id,
    r.submitted_at
FROM reviews r
JOIN pull_requests pr
    ON pr.pull_request_id = r.pull_request_id
JOIN reviewer_eligibility re
    ON re.repository_id = pr.repository_id
    AND re.developer_id = r.reviewer_id
WHERE r.review_state = 'approved'
  AND re.can_approve = TRUE
  AND r.dismissed_at IS NULL;

CREATE VIEW unresolved_review_conversations AS
SELECT
    pr.pull_request_id,
    COUNT(rc.comment_id) AS unresolved_comment_count
FROM pull_requests pr
JOIN reviews r
    ON r.pull_request_id = pr.pull_request_id
JOIN review_comments rc
    ON rc.review_id = r.review_id
WHERE rc.is_resolved = FALSE
GROUP BY pr.pull_request_id;

CREATE VIEW merge_eligibility AS
WITH approval_counts AS (
    SELECT
        pull_request_id,
        COUNT(DISTINCT reviewer_id) AS approval_count
    FROM active_eligible_approvals
    GROUP BY pull_request_id
),
required_check_failures AS (
    SELECT
        pull_request_id,
        COUNT(*) FILTER (
            WHERE status <> 'passed'
        ) AS failed_required_checks
    FROM status_checks
    WHERE required = TRUE
    GROUP BY pull_request_id
),
unresolved_counts AS (
    SELECT
        pull_request_id,
        unresolved_comment_count
    FROM unresolved_review_conversations
)
SELECT
    pr.pull_request_id,
    pr.repository_id,
    pr.number,
    pr.state,
    pr.branch_synchronized,
    pr.merge_conflict,
    COALESCE(ac.approval_count, 0) AS approval_count,
    COALESCE(rc.unresolved_comment_count, 0) AS unresolved_comments,
    COALESCE(sc.failed_required_checks, 0) AS failed_required_checks,
    rp.required_approvals,
    rp.require_status_checks,
    rp.require_conversation_resolution,
    rs.probability AS predicted_defect_probability,
    CASE
        WHEN pr.state <> 'open'
            THEN FALSE
        WHEN pr.merge_conflict
            THEN FALSE
        WHEN NOT pr.branch_synchronized
            THEN FALSE
        WHEN COALESCE(ac.approval_count, 0) < rp.required_approvals
            THEN FALSE
        WHEN rp.require_status_checks
             AND COALESCE(sc.failed_required_checks, 0) > 0
            THEN FALSE
        WHEN rp.require_conversation_resolution
             AND COALESCE(rc.unresolved_comment_count, 0) > 0
            THEN FALSE
        ELSE TRUE
    END AS policy_eligible
FROM pull_requests pr
JOIN branches target_branch
    ON target_branch.branch_id = pr.target_branch_id
JOIN branch_protection_policies rp
    ON rp.branch_id = target_branch.branch_id
LEFT JOIN approval_counts ac
    ON ac.pull_request_id = pr.pull_request_id
LEFT JOIN required_check_failures sc
    ON sc.pull_request_id = pr.pull_request_id
LEFT JOIN unresolved_counts rc
    ON rc.pull_request_id = pr.pull_request_id
LEFT JOIN logistic_risk_scores rs
    ON rs.pull_request_id = pr.pull_request_id;

INSERT INTO repositories (
    repository_name,
    default_branch
)
VALUES (
    'payments-platform',
    'main'
);

INSERT INTO developers (
    username,
    display_name
)
VALUES
    ('maya', 'Maya Singh'),
    ('rohan', 'Rohan Mehta'),
    ('elena', 'Elena Garcia'),
    ('liam', 'Liam Chen'),
    ('observer', 'Repository Observer');

INSERT INTO branches (
    repository_id,
    branch_name,
    is_protected
)
SELECT
    repository_id,
    branch_name,
    branch_name = 'main'
FROM repositories,
LATERAL (
    VALUES
        ('main'),
        ('feature/payment-retry'),
        ('feature/reporting-refactor')
) AS branch_names(branch_name);

INSERT INTO branch_protection_policies (
    branch_id,
    required_approvals,
    require_status_checks,
    require_conversation_resolution,
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
    FALSE
FROM branches
WHERE branch_name = 'main';

INSERT INTO approval_policies (
    repository_id,
    policy_name,
    minimum_approvals,
    require_unique_reviewers,
    dismiss_stale_approvals
)
SELECT
    repository_id,
    'production-merge-review',
    2,
    TRUE,
    TRUE
FROM repositories;

INSERT INTO reviewer_eligibility (
    repository_id,
    developer_id,
    can_approve
)
SELECT
    r.repository_id,
    d.developer_id,
    d.username IN ('maya', 'rohan', 'elena')
FROM repositories r
CROSS JOIN developers d;

INSERT INTO pull_requests (
    repository_id,
    number,
    source_branch_id,
    target_branch_id,
    author_id,
    title,
    state,
    is_draft,
    changed_files,
    additions,
    deletions,
    commit_count,
    branch_synchronized,
    merge_conflict,
    conversations_resolved
)
SELECT
    r.repository_id,
    1427,
    source.branch_id,
    target.branch_id,
    author.developer_id,
    'Make payment retries idempotent',
    'open',
    FALSE,
    17,
    290,
    95,
    6,
    TRUE,
    FALSE,
    TRUE
FROM repositories r
JOIN branches source
    ON source.repository_id = r.repository_id
    AND source.branch_name = 'feature/payment-retry'
JOIN branches target
    ON target.repository_id = r.repository_id
    AND target.branch_name = 'main'
JOIN developers author
    ON author.username = 'maya';

INSERT INTO pull_requests (
    repository_id,
    number,
    source_branch_id,
    target_branch_id,
    author_id,
    title,
    state,
    is_draft,
    changed_files,
    additions,
    deletions,
    commit_count,
    branch_synchronized,
    merge_conflict,
    conversations_resolved
)
SELECT
    r.repository_id,
    1451,
    source.branch_id,
    target.branch_id,
    author.developer_id,
    'Refactor reporting pipeline',
    'open',
    FALSE,
    190,
    3600,
    2100,
    35,
    FALSE,
    TRUE,
    FALSE
FROM repositories r
JOIN branches source
    ON source.repository_id = r.repository_id
    AND source.branch_name = 'feature/reporting-refactor'
JOIN branches target
    ON target.repository_id = r.repository_id
    AND target.branch_name = 'main'
JOIN developers author
    ON author.username = 'maya';

INSERT INTO commits (
    pull_request_id,
    commit_hash,
    author_id,
    message
)
SELECT
    pr.pull_request_id,
    'a1427c001',
    d.developer_id,
    'Add idempotency key to retry workflow'
FROM pull_requests pr
JOIN developers d ON d.username = 'maya'
WHERE pr.number = 1427;

INSERT INTO commits (
    pull_request_id,
    commit_hash,
    author_id,
    message
)
SELECT
    pr.pull_request_id,
    'a1427c002',
    d.developer_id,
    'Cover retry timeout behavior'
FROM pull_requests pr
JOIN developers d ON d.username = 'maya'
WHERE pr.number = 1427;

INSERT INTO reviews (
    pull_request_id,
    reviewer_id,
    review_state
)
SELECT
    pr.pull_request_id,
    d.developer_id,
    'approved'
FROM pull_requests pr
JOIN developers d
    ON d.username IN ('rohan', 'elena')
WHERE pr.number = 1427;

INSERT INTO reviews (
    pull_request_id,
    reviewer_id,
    review_state
)
SELECT
    pr.pull_request_id,
    d.developer_id,
    'commented'
FROM pull_requests pr
JOIN developers d
    ON d.username = 'observer'
WHERE pr.number = 1427;

INSERT INTO reviews (
    pull_request_id,
    reviewer_id,
    review_state,
    dismissed_at,
    dismissal_reason
)
SELECT
    pr.pull_request_id,
    d.developer_id,
    'dismissed',
    CURRENT_TIMESTAMP,
    'Approval was stale after a material change.'
FROM pull_requests pr
JOIN developers d
    ON d.username = 'rohan'
WHERE pr.number = 1451;

INSERT INTO review_comments (
    review_id,
    author_id,
    file_path,
    line_number,
    body,
    is_resolved
)
SELECT
    r.review_id,
    d.developer_id,
    'src/reporting/aggregation.cpp',
    88,
    'The aggregation lock scope is too broad.',
    FALSE
FROM reviews r
JOIN pull_requests pr
    ON pr.pull_request_id = r.pull_request_id
JOIN developers d
    ON d.username = 'rohan'
WHERE pr.number = 1451
  AND r.review_state = 'dismissed';

INSERT INTO status_checks (
    pull_request_id,
    check_name,
    status,
    required,
    completed_at
)
SELECT
    pull_request_id,
    'unit-tests',
    'passed',
    TRUE,
    CURRENT_TIMESTAMP
FROM pull_requests
WHERE number IN (1427, 1451);

INSERT INTO status_checks (
    pull_request_id,
    check_name,
    status,
    required,
    completed_at
)
SELECT
    pull_request_id,
    'integration-tests',
    CASE
        WHEN number = 1427 THEN 'passed'
        ELSE 'failed'
    END,
    TRUE,
    CURRENT_TIMESTAMP
FROM pull_requests
WHERE number IN (1427, 1451);

INSERT INTO status_checks (
    pull_request_id,
    check_name,
    status,
    required,
    completed_at
)
SELECT
    pull_request_id,
    'security-scan',
    'passed',
    TRUE,
    CURRENT_TIMESTAMP
FROM pull_requests
WHERE number IN (1427, 1451);

INSERT INTO logistic_risk_scores (
    pull_request_id,
    model_version,
    logit_value,
    probability,
    classification_threshold
)
SELECT
    pull_request_id,
    'logistic-risk-v2',
    CASE
        WHEN number = 1427 THEN -2.25
        ELSE 1.85
    END,
    CASE
        WHEN number = 1427 THEN 1.0 / (1.0 + exp(2.25))
        ELSE 1.0 / (1.0 + exp(-1.85))
    END,
    0.70
FROM pull_requests
WHERE number IN (1427, 1451);

-- A Pull Request requires two distinct eligible approvals.
SELECT
    pr.number,
    COUNT(DISTINCT aa.reviewer_id) AS eligible_approvals
FROM pull_requests pr
LEFT JOIN active_eligible_approvals aa
    ON aa.pull_request_id = pr.pull_request_id
GROUP BY pr.number
ORDER BY pr.number;

-- Review discussions are distinct from approval decisions.
-- A Pull Request can have approvals while still containing unresolved
-- review comments that branch protection forbids from remaining open.
SELECT
    pr.number,
    COALESCE(ur.unresolved_comment_count, 0)
        AS unresolved_review_comments
FROM pull_requests pr
LEFT JOIN unresolved_review_conversations ur
    ON ur.pull_request_id = pr.pull_request_id
ORDER BY pr.number;

-- Status checks are evaluated independently from reviewer decisions.
SELECT
    pr.number,
    sc.check_name,
    sc.required,
    sc.status
FROM pull_requests pr
JOIN status_checks sc
    ON sc.pull_request_id = pr.pull_request_id
ORDER BY pr.number, sc.check_name;

-- Policy-level eligibility.
SELECT
    number,
    approval_count,
    unresolved_comments,
    failed_required_checks,
    required_approvals,
    predicted_defect_probability,
    policy_eligible
FROM merge_eligibility
ORDER BY number;

-- Final merge eligibility adds the model risk threshold to explicit
-- branch-protection conditions.
SELECT
    number,
    policy_eligible,
    predicted_defect_probability,
    (
        policy_eligible
        AND predicted_defect_probability < 0.70
    ) AS merge_eligible
FROM merge_eligibility
ORDER BY number;

-- Transactional demonstration:
-- A merge attempt is recorded only when the current database state satisfies
-- the repository policy and the risk threshold.
BEGIN;

WITH eligibility AS (
    SELECT *
    FROM merge_eligibility
    WHERE number = 1427
)
INSERT INTO merge_attempts (
    pull_request_id,
    attempted_by,
    eligible,
    blocked_reason
)
SELECT
    e.pull_request_id,
    d.developer_id,
    (
        e.policy_eligible
        AND e.predicted_defect_probability < 0.70
    ),
    CASE
        WHEN NOT e.policy_eligible
            THEN 'Branch-protection policy is not satisfied.'
        WHEN e.predicted_defect_probability >= 0.70
            THEN 'Risk threshold exceeded.'
        ELSE NULL
    END
FROM eligibility e
JOIN developers d
    ON d.username = 'maya';

COMMIT;

-- Governance audit:
-- show all merge attempts with their associated Pull Request and risk score.
SELECT
    ma.merge_attempt_id,
    pr.number AS pull_request_number,
    ma.eligible,
    ma.blocked_reason,
    rs.model_version,
    rs.probability
FROM merge_attempts ma
JOIN pull_requests pr
    ON pr.pull_request_id = ma.pull_request_id
LEFT JOIN logistic_risk_scores rs
    ON rs.pull_request_id = pr.pull_request_id
ORDER BY ma.attempted_at;
