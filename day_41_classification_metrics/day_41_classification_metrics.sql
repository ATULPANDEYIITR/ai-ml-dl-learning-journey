DROP SCHEMA IF EXISTS classification_metrics_lab CASCADE;
CREATE SCHEMA classification_metrics_lab;
SET search_path TO classification_metrics_lab;

-- PostgreSQL model for evaluating classification metrics in a repository
-- governance context. The positive class means "specialist review required".

CREATE TYPE pull_request_state AS ENUM (
    'draft',
    'open',
    'changes_requested',
    'approved',
    'merged',
    'closed'
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

CREATE TABLE developers (
    developer_id BIGSERIAL PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE pull_requests (
    pull_request_id BIGSERIAL PRIMARY KEY,
    repository_id BIGINT NOT NULL REFERENCES repositories(repository_id),
    source_branch_id BIGINT NOT NULL REFERENCES branches(branch_id),
    target_branch_id BIGINT NOT NULL REFERENCES branches(branch_id),
    author_id BIGINT NOT NULL REFERENCES developers(developer_id),
    title TEXT NOT NULL,
    state pull_request_state NOT NULL DEFAULT 'open',
    is_draft BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    merged_at TIMESTAMPTZ,

    CHECK (source_branch_id <> target_branch_id),
    CHECK (
        (state = 'merged' AND merged_at IS NOT NULL)
        OR state <> 'merged'
    )
);

CREATE TABLE commits (
    commit_id BIGSERIAL PRIMARY KEY,
    pull_request_id BIGINT NOT NULL REFERENCES pull_requests(pull_request_id),
    sha CHAR(40) NOT NULL UNIQUE,
    author_id BIGINT NOT NULL REFERENCES developers(developer_id),
    message TEXT NOT NULL,
    committed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE reviewers (
    reviewer_id BIGSERIAL PRIMARY KEY,
    developer_id BIGINT NOT NULL UNIQUE REFERENCES developers(developer_id),
    can_approve BOOLEAN NOT NULL DEFAULT TRUE,
    security_reviewer BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE reviews (
    review_id BIGSERIAL PRIMARY KEY,
    pull_request_id BIGINT NOT NULL REFERENCES pull_requests(pull_request_id),
    reviewer_id BIGINT NOT NULL REFERENCES reviewers(reviewer_id),
    state review_state NOT NULL,
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    dismissed_at TIMESTAMPTZ
);

CREATE TABLE review_comments (
    comment_id BIGSERIAL PRIMARY KEY,
    review_id BIGINT NOT NULL REFERENCES reviews(review_id),
    file_path TEXT NOT NULL,
    line_number INTEGER NOT NULL,
    body TEXT NOT NULL,
    resolved BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CHECK (line_number > 0)
);

CREATE TABLE status_checks (
    status_check_id BIGSERIAL PRIMARY KEY,
    pull_request_id BIGINT NOT NULL REFERENCES pull_requests(pull_request_id),
    check_name TEXT NOT NULL,
    passed BOOLEAN NOT NULL,
    completed_at TIMESTAMPTZ,
    UNIQUE (pull_request_id, check_name)
);

CREATE TABLE branch_protection_policies (
    policy_id BIGSERIAL PRIMARY KEY,
    branch_id BIGINT NOT NULL UNIQUE REFERENCES branches(branch_id),
    required_approvals INTEGER NOT NULL DEFAULT 0,
    require_conversation_resolution BOOLEAN NOT NULL DEFAULT TRUE,
    require_status_checks BOOLEAN NOT NULL DEFAULT TRUE,
    restrict_direct_push BOOLEAN NOT NULL DEFAULT TRUE,
    restrict_force_push BOOLEAN NOT NULL DEFAULT TRUE,
    restrict_deletion BOOLEAN NOT NULL DEFAULT TRUE,
    require_linear_history BOOLEAN NOT NULL DEFAULT FALSE,
    allow_administrator_bypass BOOLEAN NOT NULL DEFAULT FALSE,

    CHECK (required_approvals >= 0)
);

CREATE TABLE approval_requirements (
    requirement_id BIGSERIAL PRIMARY KEY,
    policy_id BIGINT NOT NULL REFERENCES branch_protection_policies(policy_id),
    requirement_name TEXT NOT NULL,
    minimum_approvals INTEGER NOT NULL,
    security_review_required BOOLEAN NOT NULL DEFAULT FALSE,

    CHECK (minimum_approvals >= 0)
);

CREATE TABLE classification_results (
    result_id BIGSERIAL PRIMARY KEY,
    pull_request_id BIGINT NOT NULL REFERENCES pull_requests(pull_request_id),
    actual_label BOOLEAN NOT NULL,
    predicted_label BOOLEAN NOT NULL,
    positive_probability NUMERIC(6,5),
    threshold NUMERIC(6,5) NOT NULL DEFAULT 0.50000,
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CHECK (positive_probability IS NULL OR
           positive_probability BETWEEN 0 AND 1),
    CHECK (threshold BETWEEN 0 AND 1)
);

CREATE INDEX idx_pull_requests_target_state
    ON pull_requests(target_branch_id, state);

CREATE INDEX idx_reviews_pr_state
    ON reviews(pull_request_id, state);

CREATE INDEX idx_comments_unresolved
    ON review_comments(review_id)
    WHERE resolved = FALSE;

CREATE INDEX idx_status_checks_pr_passed
    ON status_checks(pull_request_id, passed);

CREATE INDEX idx_classification_results_labels
    ON classification_results(actual_label, predicted_label);

INSERT INTO repositories (repository_name, default_branch)
VALUES ('secure-payments-service', 'main');

INSERT INTO branches (repository_id, branch_name, is_protected)
SELECT repository_id, 'main', TRUE
FROM repositories
WHERE repository_name = 'secure-payments-service';

INSERT INTO branches (repository_id, branch_name, is_protected)
SELECT repository_id, 'feature/risk-classifier', FALSE
FROM repositories
WHERE repository_name = 'secure-payments-service';

INSERT INTO developers (username, display_name)
VALUES
    ('alice', 'Alice Developer'),
    ('bob', 'Bob Maintainer'),
    ('carol', 'Carol Security Reviewer'),
    ('david', 'David Reviewer');

INSERT INTO reviewers (developer_id, can_approve, security_reviewer)
SELECT developer_id, TRUE, FALSE
FROM developers
WHERE username = 'bob';

INSERT INTO reviewers (developer_id, can_approve, security_reviewer)
SELECT developer_id, TRUE, TRUE
FROM developers
WHERE username = 'carol';

INSERT INTO reviewers (developer_id, can_approve, security_reviewer)
SELECT developer_id, TRUE, FALSE
FROM developers
WHERE username = 'david';

INSERT INTO pull_requests (
    repository_id,
    source_branch_id,
    target_branch_id,
    author_id,
    title,
    state,
    is_draft
)
SELECT
    r.repository_id,
    source.branch_id,
    target.branch_id,
    author.developer_id,
    'Harden payment token validation',
    'open',
    FALSE
FROM repositories r
JOIN branches source
    ON source.repository_id = r.repository_id
   AND source.branch_name = 'feature/risk-classifier'
JOIN branches target
    ON target.repository_id = r.repository_id
   AND target.branch_name = 'main'
JOIN developers author
    ON author.username = 'alice'
WHERE r.repository_name = 'secure-payments-service';

INSERT INTO commits (
    pull_request_id,
    sha,
    author_id,
    message
)
SELECT
    pr.pull_request_id,
    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    d.developer_id,
    'Validate payment token issuer before authorization'
FROM pull_requests pr
JOIN developers d ON d.username = 'alice';

INSERT INTO branch_protection_policies (
    branch_id,
    required_approvals,
    require_conversation_resolution,
    require_status_checks,
    restrict_direct_push,
    restrict_force_push,
    restrict_deletion,
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

INSERT INTO approval_requirements (
    policy_id,
    requirement_name,
    minimum_approvals,
    security_review_required
)
SELECT
    policy_id,
    'Protected production change',
    2,
    TRUE
FROM branch_protection_policies
JOIN branches ON branches.branch_id =
    branch_protection_policies.branch_id
WHERE branches.branch_name = 'main';

INSERT INTO reviews (
    pull_request_id,
    reviewer_id,
    state
)
SELECT
    pr.pull_request_id,
    reviewer.reviewer_id,
    'approved'
FROM pull_requests pr
JOIN reviewers reviewer
    ON reviewer.developer_id = (
        SELECT developer_id
        FROM developers
        WHERE username = 'carol'
    );

INSERT INTO reviews (
    pull_request_id,
    reviewer_id,
    state
)
SELECT
    pr.pull_request_id,
    reviewer.reviewer_id,
    'approved'
FROM pull_requests pr
JOIN reviewers reviewer
    ON reviewer.developer_id = (
        SELECT developer_id
        FROM developers
        WHERE username = 'bob'
    );

INSERT INTO review_comments (
    review_id,
    file_path,
    line_number,
    body,
    resolved
)
SELECT
    review_id,
    'src/token_validator.cpp',
    87,
    'Validate the issuer before accepting the token.',
    TRUE
FROM reviews
WHERE pull_request_id = (
    SELECT pull_request_id
    FROM pull_requests
    ORDER BY pull_request_id
    LIMIT 1
)
AND state = 'approved'
LIMIT 1;

INSERT INTO status_checks (
    pull_request_id,
    check_name,
    passed,
    completed_at
)
SELECT
    pull_request_id,
    check_name,
    TRUE,
    CURRENT_TIMESTAMP
FROM pull_requests,
     (VALUES
        ('unit-tests'),
        ('security-scan'),
        ('build')
     ) AS checks(check_name);

INSERT INTO classification_results (
    pull_request_id,
    actual_label,
    predicted_label,
    positive_probability,
    threshold
)
SELECT
    pull_request_id,
    actual_label,
    predicted_label,
    positive_probability,
    threshold
FROM pull_requests,
     (VALUES
        (TRUE,  TRUE,  0.92::NUMERIC, 0.50::NUMERIC),
        (FALSE, FALSE, 0.12::NUMERIC, 0.50::NUMERIC),
        (TRUE,  FALSE, 0.44::NUMERIC, 0.50::NUMERIC),
        (FALSE, TRUE,  0.73::NUMERIC, 0.50::NUMERIC),
        (TRUE,  TRUE,  0.81::NUMERIC, 0.50::NUMERIC),
        (FALSE, FALSE, 0.18::NUMERIC, 0.50::NUMERIC)
     ) AS samples(
        actual_label,
        predicted_label,
        positive_probability,
        threshold
     )
WHERE pull_request_id = (
    SELECT pull_request_id
    FROM pull_requests
    ORDER BY pull_request_id
    LIMIT 1
);

-- Binary confusion matrix.

SELECT
    COUNT(*) FILTER (
        WHERE actual_label = TRUE
          AND predicted_label = TRUE
    ) AS true_positive,
    COUNT(*) FILTER (
        WHERE actual_label = FALSE
          AND predicted_label = TRUE
    ) AS false_positive,
    COUNT(*) FILTER (
        WHERE actual_label = FALSE
          AND predicted_label = FALSE
    ) AS true_negative,
    COUNT(*) FILTER (
        WHERE actual_label = TRUE
          AND predicted_label = FALSE
    ) AS false_negative
FROM classification_results;

-- Accuracy, precision, recall, specificity, and F1 are calculated from
-- the same four confusion-matrix counts. NULLIF prevents division by zero.

WITH counts AS (
    SELECT
        COUNT(*) FILTER (
            WHERE actual_label AND predicted_label
        )::NUMERIC AS tp,
        COUNT(*) FILTER (
            WHERE NOT actual_label AND predicted_label
        )::NUMERIC AS fp,
        COUNT(*) FILTER (
            WHERE NOT actual_label AND NOT predicted_label
        )::NUMERIC AS tn,
        COUNT(*) FILTER (
            WHERE actual_label AND NOT predicted_label
        )::NUMERIC AS fn
    FROM classification_results
)
SELECT
    (tp + tn) / NULLIF(tp + fp + tn + fn, 0) AS accuracy,
    tp / NULLIF(tp + fp, 0) AS precision,
    tp / NULLIF(tp + fn, 0) AS recall,
    tn / NULLIF(tn + fp, 0) AS specificity,
    (
        2 * (tp / NULLIF(tp + fp, 0))
          * (tp / NULLIF(tp + fn, 0))
    ) / NULLIF(
        (tp / NULLIF(tp + fp, 0))
        + (tp / NULLIF(tp + fn, 0)),
        0
    ) AS f1
FROM counts;

-- Threshold analysis shows how changing the decision threshold changes
-- the classification metrics without changing the underlying probabilities.

WITH thresholds(threshold) AS (
    VALUES
        (0.30::NUMERIC),
        (0.50::NUMERIC),
        (0.70::NUMERIC)
),
classified AS (
    SELECT
        t.threshold,
        r.actual_label,
        (r.positive_probability >= t.threshold) AS predicted_label
    FROM classification_results r
    CROSS JOIN thresholds t
),
counts AS (
    SELECT
        threshold,
        COUNT(*) FILTER (
            WHERE actual_label AND predicted_label
        )::NUMERIC AS tp,
        COUNT(*) FILTER (
            WHERE NOT actual_label AND predicted_label
        )::NUMERIC AS fp,
        COUNT(*) FILTER (
            WHERE NOT actual_label AND NOT predicted_label
        )::NUMERIC AS tn,
        COUNT(*) FILTER (
            WHERE actual_label AND NOT predicted_label
        )::NUMERIC AS fn
    FROM classified
    GROUP BY threshold
)
SELECT
    threshold,
    (tp + tn) / NULLIF(tp + fp + tn + fn, 0) AS accuracy,
    tp / NULLIF(tp + fp, 0) AS precision,
    tp / NULLIF(tp + fn, 0) AS recall,
    tn / NULLIF(tn + fp, 0) AS specificity,
    (
        2 * tp / NULLIF(2 * tp + fp + fn, 0)
    ) AS f1
FROM counts
ORDER BY threshold;

-- Merge eligibility is evaluated at the database layer by combining
-- branch protection, approval, status-check, and review-comment state.

WITH target AS (
    SELECT
        pr.pull_request_id,
        pr.state,
        pr.is_draft,
        bp.required_approvals,
        bp.require_status_checks,
        bp.require_conversation_resolution
    FROM pull_requests pr
    JOIN branch_protection_policies bp
        ON bp.branch_id = pr.target_branch_id
),
approval_counts AS (
    SELECT
        pull_request_id,
        COUNT(*) FILTER (
            WHERE state = 'approved'
        ) AS active_approvals
    FROM reviews
    GROUP BY pull_request_id
),
check_results AS (
    SELECT
        pull_request_id,
        BOOL_AND(passed) AS all_checks_passed
    FROM status_checks
    GROUP BY pull_request_id
),
conversation_results AS (
    SELECT
        r.pull_request_id,
        COUNT(*) FILTER (
            WHERE c.resolved = FALSE
        ) AS unresolved_comments
    FROM reviews r
    LEFT JOIN review_comments c
        ON c.review_id = r.review_id
    GROUP BY r.pull_request_id
)
SELECT
    target.pull_request_id,
    target.state,
    target.required_approvals,
    COALESCE(approval_counts.active_approvals, 0) AS active_approvals,
    COALESCE(check_results.all_checks_passed, FALSE) AS all_checks_passed,
    COALESCE(conversation_results.unresolved_comments, 0)
        AS unresolved_comments,
    (
        target.state NOT IN ('draft', 'closed', 'merged')
        AND NOT target.is_draft
        AND COALESCE(approval_counts.active_approvals, 0)
            >= target.required_approvals
        AND (
            NOT target.require_status_checks
            OR COALESCE(check_results.all_checks_passed, FALSE)
        )
        AND (
            NOT target.require_conversation_resolution
            OR COALESCE(conversation_results.unresolved_comments, 0) = 0
        )
    ) AS merge_eligible
FROM target
LEFT JOIN approval_counts
    ON approval_counts.pull_request_id = target.pull_request_id
LEFT JOIN check_results
    ON check_results.pull_request_id = target.pull_request_id
LEFT JOIN conversation_results
    ON conversation_results.pull_request_id = target.pull_request_id;

-- Demonstrate an integrity failure: a negative approval requirement
-- violates the CHECK constraint.

DO $$
BEGIN
    BEGIN
        INSERT INTO branch_protection_policies (
            branch_id,
            required_approvals
        )
        SELECT branch_id, -1
        FROM branches
        WHERE branch_name = 'main';

        RAISE EXCEPTION 'Expected CHECK constraint failure did not occur.';
    EXCEPTION
        WHEN check_violation THEN
            RAISE NOTICE 'Invalid negative approval requirement rejected.';
    END;
END
$$;

-- A transaction demonstrates atomic status updates. If any operation fails,
-- the transaction can be rolled back so the repository state remains
-- internally consistent.

BEGIN;

UPDATE pull_requests
SET state = 'approved'
WHERE pull_request_id = (
    SELECT MIN(pull_request_id)
    FROM pull_requests
);

INSERT INTO status_checks (
    pull_request_id,
    check_name,
    passed,
    completed_at
)
SELECT
    MIN(pull_request_id),
    'classification-policy-check',
    TRUE,
    CURRENT_TIMESTAMP
FROM pull_requests;

COMMIT;

-- Final operational view combines classifier performance with repository
-- governance state for auditing.

CREATE OR REPLACE VIEW classification_governance_audit AS
WITH metrics AS (
    SELECT
        COUNT(*) FILTER (
            WHERE actual_label AND predicted_label
        ) AS tp,
        COUNT(*) FILTER (
            WHERE NOT actual_label AND predicted_label
        ) AS fp,
        COUNT(*) FILTER (
            WHERE NOT actual_label AND NOT predicted_label
        ) AS tn,
        COUNT(*) FILTER (
            WHERE actual_label AND NOT predicted_label
        ) AS fn
    FROM classification_results
)
SELECT
    tp,
    fp,
    tn,
    fn,
    (tp + tn)::NUMERIC / NULLIF(tp + fp + tn + fn, 0) AS accuracy,
    tp::NUMERIC / NULLIF(tp + fp, 0) AS precision,
    tp::NUMERIC / NULLIF(tp + fn, 0) AS recall,
    tn::NUMERIC / NULLIF(tn + fp, 0) AS specificity,
    2 * tp::NUMERIC / NULLIF(2 * tp + fp + fn, 0) AS f1
FROM metrics;

SELECT *
FROM classification_governance_audit;
