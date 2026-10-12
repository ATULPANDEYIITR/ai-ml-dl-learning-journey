-- PostgreSQL 15+ compatible
-- ROC, Precision-Recall, AUC, threshold optimization, and repository
-- governance data model.
--
-- The statistical model represents a classifier that predicts whether a
-- Pull Request requires additional review attention. Repository policy remains
-- an independent governance layer.

DROP SCHEMA IF EXISTS roc_pr_governance CASCADE;
CREATE SCHEMA roc_pr_governance;
SET search_path TO roc_pr_governance;

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

CREATE TABLE repositories (
    repository_id BIGSERIAL PRIMARY KEY,
    repository_name TEXT NOT NULL UNIQUE,
    default_branch TEXT NOT NULL DEFAULT 'main',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE branches (
    branch_id BIGSERIAL PRIMARY KEY,
    repository_id BIGINT NOT NULL REFERENCES repositories(repository_id),
    branch_name TEXT NOT NULL,
    is_protected BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (repository_id, branch_name)
);

CREATE TABLE users (
    user_id BIGSERIAL PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE pull_requests (
    pull_request_id BIGSERIAL PRIMARY KEY,
    repository_id BIGINT NOT NULL REFERENCES repositories(repository_id),
    source_branch_id BIGINT NOT NULL REFERENCES branches(branch_id),
    target_branch_id BIGINT NOT NULL REFERENCES branches(branch_id),
    author_id BIGINT NOT NULL REFERENCES users(user_id),
    state pull_request_state NOT NULL DEFAULT 'draft',
    is_draft BOOLEAN NOT NULL DEFAULT TRUE,
    head_sha TEXT NOT NULL,
    base_sha TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (source_branch_id <> target_branch_id)
);

CREATE TABLE commits (
    commit_sha TEXT PRIMARY KEY,
    repository_id BIGINT NOT NULL REFERENCES repositories(repository_id),
    author_id BIGINT NOT NULL REFERENCES users(user_id),
    committed_at TIMESTAMPTZ NOT NULL,
    message TEXT NOT NULL
);

CREATE TABLE pull_request_commits (
    pull_request_id BIGINT NOT NULL REFERENCES pull_requests(pull_request_id),
    commit_sha TEXT NOT NULL REFERENCES commits(commit_sha),
    PRIMARY KEY (pull_request_id, commit_sha)
);

CREATE TABLE reviews (
    review_id BIGSERIAL PRIMARY KEY,
    pull_request_id BIGINT NOT NULL REFERENCES pull_requests(pull_request_id),
    reviewer_id BIGINT NOT NULL REFERENCES users(user_id),
    state review_state NOT NULL,
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    commit_sha TEXT NOT NULL,
    body TEXT
);

CREATE TABLE review_comments (
    comment_id BIGSERIAL PRIMARY KEY,
    review_id BIGINT NOT NULL REFERENCES reviews(review_id),
    author_id BIGINT NOT NULL REFERENCES users(user_id),
    file_path TEXT NOT NULL,
    line_number INTEGER CHECK (line_number > 0),
    body TEXT NOT NULL,
    resolved BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE status_checks (
    status_check_id BIGSERIAL PRIMARY KEY,
    pull_request_id BIGINT NOT NULL REFERENCES pull_requests(pull_request_id),
    check_name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'success', 'failure', 'cancelled')
    ),
    completed_at TIMESTAMPTZ,
    UNIQUE (pull_request_id, check_name)
);

CREATE TABLE branch_protection_policies (
    policy_id BIGSERIAL PRIMARY KEY,
    branch_id BIGINT NOT NULL UNIQUE REFERENCES branches(branch_id),
    required_approvals INTEGER NOT NULL DEFAULT 1 CHECK (required_approvals >= 0),
    require_conversation_resolution BOOLEAN NOT NULL DEFAULT TRUE,
    require_status_checks BOOLEAN NOT NULL DEFAULT TRUE,
    restrict_direct_push BOOLEAN NOT NULL DEFAULT TRUE,
    allow_force_push BOOLEAN NOT NULL DEFAULT FALSE,
    allow_branch_deletion BOOLEAN NOT NULL DEFAULT FALSE,
    require_linear_history BOOLEAN NOT NULL DEFAULT FALSE,
    dismiss_stale_approvals BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE approval_requirements (
    requirement_id BIGSERIAL PRIMARY KEY,
    policy_id BIGINT NOT NULL REFERENCES branch_protection_policies(policy_id),
    required_team TEXT,
    required_user_id BIGINT REFERENCES users(user_id),
    CHECK (
        required_team IS NOT NULL OR required_user_id IS NOT NULL
    )
);

CREATE TABLE model_scores (
    score_id BIGSERIAL PRIMARY KEY,
    pull_request_id BIGINT NOT NULL REFERENCES pull_requests(pull_request_id),
    evaluation_split TEXT NOT NULL CHECK (
        evaluation_split IN ('train', 'validation', 'test', 'production')
    ),
    actual_label INTEGER CHECK (actual_label IN (0, 1)),
    predicted_score NUMERIC(10, 8) NOT NULL CHECK (
        predicted_score >= 0 AND predicted_score <= 1
    ),
    model_version TEXT NOT NULL,
    scored_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_pr_repository_target
    ON pull_requests(repository_id, target_branch_id, state);

CREATE INDEX idx_reviews_pr_state
    ON reviews(pull_request_id, state);

CREATE INDEX idx_status_checks_pr_status
    ON status_checks(pull_request_id, status);

CREATE INDEX idx_model_scores_split_score
    ON model_scores(evaluation_split, predicted_score DESC);

CREATE VIEW effective_review_decisions AS
SELECT
    r.pull_request_id,
    r.reviewer_id,
    r.review_id,
    r.state,
    r.commit_sha,
    r.submitted_at
FROM reviews r
JOIN pull_requests pr
  ON pr.pull_request_id = r.pull_request_id
WHERE r.state <> 'dismissed'
  AND r.commit_sha = pr.head_sha;

INSERT INTO repositories (repository_name)
VALUES ('payments-api');

INSERT INTO users (username)
VALUES
    ('atul'),
    ('reviewer_a'),
    ('reviewer_b'),
    ('security_reviewer');

INSERT INTO branches (
    repository_id,
    branch_name,
    is_protected
)
SELECT repository_id, 'main', TRUE
FROM repositories
WHERE repository_name = 'payments-api';

INSERT INTO branches (
    repository_id,
    branch_name,
    is_protected
)
SELECT repository_id, 'feature/payment-risk', FALSE
FROM repositories
WHERE repository_name = 'payments-api';

INSERT INTO branch_protection_policies (
    branch_id,
    required_approvals,
    require_conversation_resolution,
    require_status_checks,
    restrict_direct_push,
    allow_force_push,
    allow_branch_deletion,
    require_linear_history,
    dismiss_stale_approvals
)
SELECT
    branch_id,
    2,
    TRUE,
    TRUE,
    TRUE,
    FALSE,
    FALSE,
    TRUE,
    TRUE
FROM branches
WHERE branch_name = 'main';

INSERT INTO approval_requirements (
    policy_id,
    required_user_id
)
SELECT
    policy_id,
    (SELECT user_id FROM users WHERE username = 'security_reviewer')
FROM branch_protection_policies
JOIN branches
  ON branches.branch_id = branch_protection_policies.branch_id
WHERE branches.branch_name = 'main';

INSERT INTO pull_requests (
    repository_id,
    source_branch_id,
    target_branch_id,
    author_id,
    state,
    is_draft,
    head_sha,
    base_sha
)
SELECT
    r.repository_id,
    source.branch_id,
    target.branch_id,
    author.user_id,
    'open',
    FALSE,
    'abc123head',
    'main789base'
FROM repositories r
JOIN branches source
  ON source.repository_id = r.repository_id
 AND source.branch_name = 'feature/payment-risk'
JOIN branches target
  ON target.repository_id = r.repository_id
 AND target.branch_name = 'main'
JOIN users author
  ON author.username = 'atul'
WHERE r.repository_name = 'payments-api';

INSERT INTO commits (
    commit_sha,
    repository_id,
    author_id,
    committed_at,
    message
)
SELECT
    'abc123head',
    repository_id,
    (SELECT user_id FROM users WHERE username = 'atul'),
    CURRENT_TIMESTAMP,
    'Add payment risk scoring'
FROM repositories
WHERE repository_name = 'payments-api';

INSERT INTO pull_request_commits (
    pull_request_id,
    commit_sha
)
SELECT pull_request_id, 'abc123head'
FROM pull_requests
WHERE head_sha = 'abc123head';

INSERT INTO status_checks (
    pull_request_id,
    check_name,
    status,
    completed_at
)
SELECT
    pull_request_id,
    check_name,
    'success',
    CURRENT_TIMESTAMP
FROM pull_requests
CROSS JOIN (
    VALUES
        ('unit-tests'),
        ('integration-tests'),
        ('security-scan')
) AS checks(check_name)
WHERE head_sha = 'abc123head';

INSERT INTO reviews (
    pull_request_id,
    reviewer_id,
    state,
    commit_sha,
    body
)
SELECT
    pr.pull_request_id,
    u.user_id,
    'approved',
    pr.head_sha,
    'Application behavior reviewed.'
FROM pull_requests pr
JOIN users u ON u.username = 'reviewer_a'
WHERE pr.head_sha = 'abc123head';

INSERT INTO reviews (
    pull_request_id,
    reviewer_id,
    state,
    commit_sha,
    body
)
SELECT
    pr.pull_request_id,
    u.user_id,
    'approved',
    pr.head_sha,
    'Security impact reviewed.'
FROM pull_requests pr
JOIN users u ON u.username = 'security_reviewer'
WHERE pr.head_sha = 'abc123head';

INSERT INTO review_comments (
    review_id,
    author_id,
    file_path,
    line_number,
    body,
    resolved
)
SELECT
    r.review_id,
    r.reviewer_id,
    'risk_engine.py',
    87,
    'Threshold selection must be validated on held-out data.',
    TRUE
FROM reviews r
WHERE r.commit_sha = 'abc123head'
LIMIT 1;

INSERT INTO model_scores (
    pull_request_id,
    evaluation_split,
    actual_label,
    predicted_score,
    model_version
)
SELECT
    pr.pull_request_id,
    sample.evaluation_split,
    sample.actual_label,
    sample.predicted_score,
    'review-risk-v2'
FROM pull_requests pr
CROSS JOIN (
    VALUES
        ('test', 0, 0.04),
        ('test', 0, 0.08),
        ('test', 0, 0.12),
        ('test', 0, 0.18),
        ('test', 0, 0.24),
        ('test', 0, 0.31),
        ('test', 0, 0.39),
        ('test', 0, 0.47),
        ('test', 0, 0.56),
        ('test', 0, 0.62),
        ('test', 1, 0.35),
        ('test', 1, 0.51),
        ('test', 1, 0.59),
        ('test', 1, 0.68),
        ('test', 1, 0.77),
        ('test', 1, 0.91)
) AS sample(evaluation_split, actual_label, predicted_score)
WHERE pr.head_sha = 'abc123head';

-- Inspect precision, recall, specificity, and FPR for every observed
-- threshold. This query makes threshold selection explicit instead of
-- pretending that AUC itself selects an operating point.
WITH thresholds AS (
    SELECT DISTINCT predicted_score AS threshold
    FROM model_scores
    WHERE evaluation_split = 'test'

    UNION ALL

    SELECT 0.0

    UNION ALL

    SELECT 1.0
),
confusion AS (
    SELECT
        t.threshold,
        COUNT(*) FILTER (
            WHERE s.actual_label = 1
              AND s.predicted_score >= t.threshold
        ) AS tp,
        COUNT(*) FILTER (
            WHERE s.actual_label = 0
              AND s.predicted_score >= t.threshold
        ) AS fp,
        COUNT(*) FILTER (
            WHERE s.actual_label = 0
              AND s.predicted_score < t.threshold
        ) AS tn,
        COUNT(*) FILTER (
            WHERE s.actual_label = 1
              AND s.predicted_score < t.threshold
        ) AS fn
    FROM thresholds t
    CROSS JOIN model_scores s
    WHERE s.evaluation_split = 'test'
    GROUP BY t.threshold
)
SELECT
    threshold,
    tp,
    fp,
    tn,
    fn,
    ROUND(
        tp::NUMERIC / NULLIF(tp + fp, 0),
        4
    ) AS precision,
    ROUND(
        tp::NUMERIC / NULLIF(tp + fn, 0),
        4
    ) AS recall,
    ROUND(
        tn::NUMERIC / NULLIF(tn + fp, 0),
        4
    ) AS specificity,
    ROUND(
        fp::NUMERIC / NULLIF(fp + tn, 0),
        4
    ) AS false_positive_rate,
    ROUND(
        2.0 * tp / NULLIF(2 * tp + fp + fn, 0),
        4
    ) AS f1
FROM confusion
ORDER BY threshold DESC;

-- ROC points and trapezoidal AUC.
WITH thresholds AS (
    SELECT DISTINCT predicted_score AS threshold
    FROM model_scores
    WHERE evaluation_split = 'test'
    UNION
    SELECT 0.0
    UNION
    SELECT 1.0
),
points AS (
    SELECT
        t.threshold,
        COUNT(*) FILTER (
            WHERE s.actual_label = 1
              AND s.predicted_score >= t.threshold
        )::NUMERIC /
        NULLIF(
            COUNT(*) FILTER (WHERE s.actual_label = 1),
            0
        ) AS tpr,
        COUNT(*) FILTER (
            WHERE s.actual_label = 0
              AND s.predicted_score >= t.threshold
        )::NUMERIC /
        NULLIF(
            COUNT(*) FILTER (WHERE s.actual_label = 0),
            0
        ) AS fpr
    FROM thresholds t
    CROSS JOIN model_scores s
    WHERE s.evaluation_split = 'test'
    GROUP BY t.threshold
),
ordered AS (
    SELECT
        *,
        LAG(fpr) OVER (ORDER BY fpr, tpr) AS previous_fpr,
        LAG(tpr) OVER (ORDER BY fpr, tpr) AS previous_tpr
    FROM points
)
SELECT
    SUM(
        (fpr - previous_fpr) *
        (tpr + previous_tpr) / 2.0
    ) AS roc_auc
FROM ordered
WHERE previous_fpr IS NOT NULL;

-- Precision-recall operating points.
WITH thresholds AS (
    SELECT DISTINCT predicted_score AS threshold
    FROM model_scores
    WHERE evaluation_split = 'test'
    UNION
    SELECT 0.0
    UNION
    SELECT 1.0
)
SELECT
    t.threshold,
    COUNT(*) FILTER (
        WHERE s.actual_label = 1
          AND s.predicted_score >= t.threshold
    )::NUMERIC /
    NULLIF(
        COUNT(*) FILTER (
            WHERE s.predicted_score >= t.threshold
        ),
        0
    ) AS precision,
    COUNT(*) FILTER (
        WHERE s.actual_label = 1
          AND s.predicted_score >= t.threshold
    )::NUMERIC /
    NULLIF(
        COUNT(*) FILTER (WHERE s.actual_label = 1),
        0
    ) AS recall
FROM thresholds t
CROSS JOIN model_scores s
WHERE s.evaluation_split = 'test'
GROUP BY t.threshold
ORDER BY recall, threshold;

-- Select the F1-optimal threshold. This is an operating decision, not an AUC
-- calculation.
WITH thresholds AS (
    SELECT DISTINCT predicted_score AS threshold
    FROM model_scores
    WHERE evaluation_split = 'test'
),
metrics AS (
    SELECT
        t.threshold,
        COUNT(*) FILTER (
            WHERE s.actual_label = 1
              AND s.predicted_score >= t.threshold
        ) AS tp,
        COUNT(*) FILTER (
            WHERE s.actual_label = 0
              AND s.predicted_score >= t.threshold
        ) AS fp,
        COUNT(*) FILTER (
            WHERE s.actual_label = 1
              AND s.predicted_score < t.threshold
        ) AS fn
    FROM thresholds t
    CROSS JOIN model_scores s
    WHERE s.evaluation_split = 'test'
    GROUP BY t.threshold
),
scored AS (
    SELECT
        *,
        2.0 * tp / NULLIF(2 * tp + fp + fn, 0) AS f1
    FROM metrics
)
SELECT
    threshold,
    tp,
    fp,
    fn,
    ROUND(f1, 4) AS f1
FROM scored
ORDER BY f1 DESC, threshold DESC
LIMIT 1;

-- Cost-sensitive threshold optimization. False negatives are five times as
-- costly as false positives, so the objective is expected error cost rather
-- than maximum F1.
WITH thresholds AS (
    SELECT DISTINCT predicted_score AS threshold
    FROM model_scores
    WHERE evaluation_split = 'test'
),
costs AS (
    SELECT
        t.threshold,
        COUNT(*) FILTER (
            WHERE s.actual_label = 0
              AND s.predicted_score >= t.threshold
        ) AS fp,
        COUNT(*) FILTER (
            WHERE s.actual_label = 1
              AND s.predicted_score < t.threshold
        ) AS fn
    FROM thresholds t
    CROSS JOIN model_scores s
    WHERE s.evaluation_split = 'test'
    GROUP BY t.threshold
)
SELECT
    threshold,
    fp,
    fn,
    fp + 5 * fn AS total_cost
FROM costs
ORDER BY total_cost, threshold DESC
LIMIT 1;

-- A constrained threshold policy: maximize recall while maintaining an FPR
-- ceiling of 10 percent.
WITH thresholds AS (
    SELECT DISTINCT predicted_score AS threshold
    FROM model_scores
    WHERE evaluation_split = 'test'
),
metrics AS (
    SELECT
        t.threshold,
        COUNT(*) FILTER (
            WHERE s.actual_label = 1
              AND s.predicted_score >= t.threshold
        )::NUMERIC /
        NULLIF(COUNT(*) FILTER (WHERE s.actual_label = 1), 0) AS recall,
        COUNT(*) FILTER (
            WHERE s.actual_label = 0
              AND s.predicted_score >= t.threshold
        )::NUMERIC /
        NULLIF(COUNT(*) FILTER (WHERE s.actual_label = 0), 0) AS fpr
    FROM thresholds t
    CROSS JOIN model_scores s
    WHERE s.evaluation_split = 'test'
    GROUP BY t.threshold
)
SELECT
    threshold,
    ROUND(recall, 4) AS recall,
    ROUND(fpr, 4) AS false_positive_rate
FROM metrics
WHERE fpr <= 0.10
ORDER BY recall DESC, threshold DESC
LIMIT 1;

-- Database-level merge eligibility. Approval counts are restricted to current
-- head_sha reviews, which prevents approvals from an older changeset from
-- automatically authorizing a newer source state.
WITH current_reviews AS (
    SELECT
        r.pull_request_id,
        COUNT(*) FILTER (
            WHERE r.state = 'approved'
        ) AS approvals
    FROM reviews r
    JOIN pull_requests pr
      ON pr.pull_request_id = r.pull_request_id
    WHERE r.commit_sha = pr.head_sha
      AND r.state <> 'dismissed'
    GROUP BY r.pull_request_id
),
check_state AS (
    SELECT
        pull_request_id,
        BOOL_AND(status = 'success') AS all_checks_passing
    FROM status_checks
    GROUP BY pull_request_id
),
unresolved AS (
    SELECT
        r.pull_request_id,
        COUNT(*) FILTER (WHERE c.resolved = FALSE) AS unresolved_count
    FROM reviews r
    LEFT JOIN review_comments c
      ON c.review_id = r.review_id
    GROUP BY r.pull_request_id
)
SELECT
    pr.pull_request_id,
    pr.state,
    COALESCE(cr.approvals, 0) AS current_approvals,
    COALESCE(cs.all_checks_passing, FALSE) AS all_checks_passing,
    COALESCE(u.unresolved_count, 0) AS unresolved_comments,
    (
        pr.state = 'open'
        AND pr.is_draft = FALSE
        AND COALESCE(cr.approvals, 0) >= bp.required_approvals
        AND COALESCE(cs.all_checks_passing, FALSE)
        AND COALESCE(u.unresolved_count, 0) = 0
    ) AS merge_eligible
FROM pull_requests pr
JOIN branch_protection_policies bp
  ON bp.branch_id = pr.target_branch_id
LEFT JOIN current_reviews cr
  ON cr.pull_request_id = pr.pull_request_id
LEFT JOIN check_state cs
  ON cs.pull_request_id = pr.pull_request_id
LEFT JOIN unresolved u
  ON u.pull_request_id = pr.pull_request_id;

-- Demonstrate transactional governance. The transaction locks the target PR
-- while checking the policy and changing its state, preventing a concurrent
-- workflow from observing a partially applied merge decision.
BEGIN;

SELECT pull_request_id
FROM pull_requests
WHERE pull_request_id = (
    SELECT MIN(pull_request_id)
    FROM pull_requests
)
FOR UPDATE;

UPDATE pull_requests
SET updated_at = CURRENT_TIMESTAMP
WHERE pull_request_id = (
    SELECT MIN(pull_request_id)
    FROM pull_requests
);

COMMIT;

-- The following query exposes positive prevalence, which is the baseline
-- precision expected from random positive retrieval.
SELECT
    COUNT(*) FILTER (WHERE actual_label = 1) AS positive_examples,
    COUNT(*) AS total_examples,
    ROUND(
        COUNT(*) FILTER (WHERE actual_label = 1)::NUMERIC /
        NULLIF(COUNT(*), 0),
        4
    ) AS positive_prevalence
FROM model_scores
WHERE evaluation_split = 'test';
