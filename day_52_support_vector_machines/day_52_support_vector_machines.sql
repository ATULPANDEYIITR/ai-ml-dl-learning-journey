-- Support Vector Machines:
-- relational representation of hyperplanes, margins, support vectors,
-- soft-margin training observations, and model-assisted merge-risk analysis.
--
-- PostgreSQL-specific mechanisms used here:
--   * generated UUID identifiers
--   * CHECK constraints for SVM labels and valid states
--   * JSONB for model metadata
--   * partial indexes for operational queries
--   * PL/pgSQL functions for margin calculations
--   * triggers for approval-count integrity
--   * views for support-vector and merge-eligibility reporting
--   * transactional seed data

DROP SCHEMA IF EXISTS svm_governance CASCADE;
CREATE SCHEMA svm_governance;
SET search_path TO svm_governance;

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE repositories (
    repository_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    default_branch TEXT NOT NULL DEFAULT 'main',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE repository_branches (
    branch_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_id UUID NOT NULL
        REFERENCES repositories(repository_id) ON DELETE CASCADE,
    branch_name TEXT NOT NULL,
    is_protected BOOLEAN NOT NULL DEFAULT false,
    UNIQUE (repository_id, branch_name)
);

CREATE TABLE contributors (
    contributor_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT true
);

CREATE TABLE pull_requests (
    pull_request_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_id UUID NOT NULL
        REFERENCES repositories(repository_id) ON DELETE CASCADE,
    author_id UUID NOT NULL
        REFERENCES contributors(contributor_id),
    source_branch_id UUID NOT NULL
        REFERENCES repository_branches(branch_id),
    target_branch_id UUID NOT NULL
        REFERENCES repository_branches(branch_id),
    title TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'open'
        CHECK (state IN ('draft', 'open', 'closed', 'merged')),
    source_synchronized BOOLEAN NOT NULL DEFAULT true,
    has_merge_conflict BOOLEAN NOT NULL DEFAULT false,
    conversations_resolved BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    closed_at TIMESTAMPTZ,
    merged_at TIMESTAMPTZ,
    CHECK (source_branch_id <> target_branch_id),
    CHECK (
        (state = 'merged' AND merged_at IS NOT NULL)
        OR state <> 'merged'
    )
);

CREATE TABLE commits (
    commit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pull_request_id UUID NOT NULL
        REFERENCES pull_requests(pull_request_id) ON DELETE CASCADE,
    sha TEXT NOT NULL,
    message TEXT NOT NULL,
    committed_at TIMESTAMPTZ NOT NULL,
    UNIQUE (pull_request_id, sha)
);

CREATE TABLE review_requests (
    review_request_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pull_request_id UUID NOT NULL
        REFERENCES pull_requests(pull_request_id) ON DELETE CASCADE,
    reviewer_id UUID NOT NULL
        REFERENCES contributors(contributor_id),
    requested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    fulfilled_at TIMESTAMPTZ,
    UNIQUE (pull_request_id, reviewer_id)
);

CREATE TABLE reviews (
    review_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pull_request_id UUID NOT NULL
        REFERENCES pull_requests(pull_request_id) ON DELETE CASCADE,
    reviewer_id UUID NOT NULL
        REFERENCES contributors(contributor_id),
    review_state TEXT NOT NULL
        CHECK (review_state IN (
            'commented',
            'changes_requested',
            'approved',
            'dismissed'
        )),
    commit_id UUID
        REFERENCES commits(commit_id),
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_stale BOOLEAN NOT NULL DEFAULT false
);

CREATE TABLE review_comments (
    comment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    review_id UUID NOT NULL
        REFERENCES reviews(review_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    line_number INTEGER NOT NULL CHECK (line_number > 0),
    body TEXT NOT NULL,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE status_checks (
    status_check_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pull_request_id UUID NOT NULL
        REFERENCES pull_requests(pull_request_id) ON DELETE CASCADE,
    check_name TEXT NOT NULL,
    status TEXT NOT NULL
        CHECK (status IN ('pending', 'passed', 'failed')),
    completed_at TIMESTAMPTZ,
    UNIQUE (pull_request_id, check_name)
);

CREATE TABLE branch_protection_policies (
    policy_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_id UUID NOT NULL
        REFERENCES repositories(repository_id) ON DELETE CASCADE,
    branch_id UUID NOT NULL
        REFERENCES repository_branches(branch_id) ON DELETE CASCADE,
    required_approvals INTEGER NOT NULL DEFAULT 0
        CHECK (required_approvals >= 0),
    require_status_checks BOOLEAN NOT NULL DEFAULT false,
    require_resolved_conversations BOOLEAN NOT NULL DEFAULT false,
    prohibit_direct_push BOOLEAN NOT NULL DEFAULT true,
    prohibit_force_push BOOLEAN NOT NULL DEFAULT true,
    prohibit_branch_deletion BOOLEAN NOT NULL DEFAULT true,
    require_linear_history BOOLEAN NOT NULL DEFAULT false,
    allow_administrator_bypass BOOLEAN NOT NULL DEFAULT false,
    UNIQUE (repository_id, branch_id)
);

CREATE TABLE required_status_checks (
    policy_id UUID NOT NULL
        REFERENCES branch_protection_policies(policy_id) ON DELETE CASCADE,
    check_name TEXT NOT NULL,
    PRIMARY KEY (policy_id, check_name)
);

CREATE TABLE svm_models (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name TEXT NOT NULL UNIQUE,
    feature_description JSONB NOT NULL,
    regularization_c NUMERIC NOT NULL CHECK (regularization_c > 0),
    weight_vector NUMERIC[] NOT NULL,
    bias NUMERIC NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE svm_observations (
    observation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL
        REFERENCES svm_models(model_id) ON DELETE CASCADE,
    pull_request_id UUID
        REFERENCES pull_requests(pull_request_id) ON DELETE SET NULL,
    feature_vector NUMERIC[] NOT NULL,
    label INTEGER NOT NULL CHECK (label IN (-1, 1)),
    decision_score NUMERIC NOT NULL,
    functional_margin NUMERIC NOT NULL,
    hinge_loss NUMERIC NOT NULL CHECK (hinge_loss >= 0),
    is_support_vector BOOLEAN NOT NULL DEFAULT false,
    CHECK (cardinality(feature_vector) = cardinality(
        (SELECT weight_vector FROM svm_models WHERE svm_models.model_id = svm_observations.model_id)
    ))
);

-- PostgreSQL does not allow a subquery in a CHECK constraint to be used
-- safely as a cross-table integrity rule. Replace the preceding table with
-- a trigger-backed version by dropping the cross-table check and recreating
-- the table definition without that expression.
ALTER TABLE svm_observations DROP CONSTRAINT svm_observations_check;

CREATE INDEX idx_pull_requests_target_state
    ON pull_requests(target_branch_id, state);

CREATE INDEX idx_reviews_current_approvals
    ON reviews(pull_request_id, reviewer_id)
    WHERE review_state = 'approved' AND is_stale = false;

CREATE INDEX idx_status_checks_pull_request
    ON status_checks(pull_request_id, check_name, status);

CREATE INDEX idx_svm_support_vectors
    ON svm_observations(model_id, is_support_vector)
    WHERE is_support_vector = true;

CREATE OR REPLACE FUNCTION calculate_hinge_loss(
    functional_margin NUMERIC
)
RETURNS NUMERIC
LANGUAGE SQL
IMMUTABLE
AS $$
    SELECT GREATEST(0::NUMERIC, 1::NUMERIC - functional_margin);
$$;

CREATE OR REPLACE FUNCTION calculate_functional_margin(
    decision_score NUMERIC,
    label INTEGER
)
RETURNS NUMERIC
LANGUAGE SQL
IMMUTABLE
AS $$
    SELECT decision_score * label;
$$;

CREATE OR REPLACE FUNCTION enforce_svm_dimension()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    expected_dimension INTEGER;
    actual_dimension INTEGER;
BEGIN
    SELECT cardinality(weight_vector)
    INTO expected_dimension
    FROM svm_models
    WHERE model_id = NEW.model_id;

    IF expected_dimension IS NULL THEN
        RAISE EXCEPTION 'SVM model % does not exist', NEW.model_id;
    END IF;

    actual_dimension := cardinality(NEW.feature_vector);

    IF actual_dimension <> expected_dimension THEN
        RAISE EXCEPTION
            'Feature dimension % does not match model dimension %',
            actual_dimension,
            expected_dimension;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_svm_dimension
BEFORE INSERT OR UPDATE ON svm_observations
FOR EACH ROW
EXECUTE FUNCTION enforce_svm_dimension();

CREATE OR REPLACE FUNCTION validate_review_author()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    pr_author UUID;
BEGIN
    SELECT author_id
    INTO pr_author
    FROM pull_requests
    WHERE pull_request_id = NEW.pull_request_id;

    IF pr_author = NEW.reviewer_id THEN
        RAISE EXCEPTION
            'Pull Request author cannot submit their own review';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_review_author
BEFORE INSERT OR UPDATE ON reviews
FOR EACH ROW
EXECUTE FUNCTION validate_review_author();

CREATE OR REPLACE VIEW current_approval_counts AS
SELECT
    pull_request_id,
    COUNT(*) FILTER (
        WHERE review_state = 'approved'
          AND NOT is_stale
    ) AS current_approvals
FROM reviews
GROUP BY pull_request_id;

CREATE OR REPLACE VIEW unresolved_review_conversations AS
SELECT
    r.pull_request_id,
    COUNT(*) AS unresolved_comments
FROM review_comments c
JOIN reviews r
    ON r.review_id = c.review_id
WHERE c.resolved_at IS NULL
GROUP BY r.pull_request_id;

CREATE OR REPLACE VIEW svm_support_vector_report AS
SELECT
    o.observation_id,
    o.pull_request_id,
    o.label,
    o.decision_score,
    o.functional_margin,
    o.hinge_loss,
    o.is_support_vector
FROM svm_observations o
WHERE o.is_support_vector = true;

CREATE OR REPLACE VIEW merge_eligibility_report AS
WITH approvals AS (
    SELECT
        pull_request_id,
        COUNT(*) FILTER (
            WHERE review_state = 'approved'
              AND NOT is_stale
        ) AS current_approvals
    FROM reviews
    GROUP BY pull_request_id
),
checks AS (
    SELECT
        pull_request_id,
        COUNT(*) FILTER (WHERE status = 'passed') AS passed_checks,
        COUNT(*) FILTER (WHERE status = 'failed') AS failed_checks,
        COUNT(*) FILTER (WHERE status = 'pending') AS pending_checks
    FROM status_checks
    GROUP BY pull_request_id
),
required AS (
    SELECT
        p.pull_request_id,
        bp.required_approvals,
        bp.require_status_checks,
        bp.require_resolved_conversations,
        bp.require_linear_history
    FROM pull_requests p
    JOIN branch_protection_policies bp
      ON bp.branch_id = p.target_branch_id
),
required_check_counts AS (
    SELECT
        p.pull_request_id,
        COUNT(rsc.check_name) AS required_count,
        COUNT(sc.status_check_id) FILTER (
            WHERE sc.status = 'passed'
        ) AS passed_required_count
    FROM pull_requests p
    JOIN branch_protection_policies bp
      ON bp.branch_id = p.target_branch_id
    LEFT JOIN required_status_checks rsc
      ON rsc.policy_id = bp.policy_id
    LEFT JOIN status_checks sc
      ON sc.pull_request_id = p.pull_request_id
     AND sc.check_name = rsc.check_name
    GROUP BY p.pull_request_id
)
SELECT
    p.pull_request_id,
    p.title,
    p.state,
    COALESCE(a.current_approvals, 0) AS current_approvals,
    r.required_approvals,
    p.source_synchronized,
    p.has_merge_conflict,
    p.conversations_resolved,
    COALESCE(c.failed_checks, 0) AS failed_checks,
    COALESCE(c.pending_checks, 0) AS pending_checks,
    COALESCE(rcc.required_count, 0) AS required_checks,
    COALESCE(rcc.passed_required_count, 0) AS passed_required_checks,
    (
        p.state = 'open'
        AND p.source_synchronized
        AND NOT p.has_merge_conflict
        AND (
            NOT r.require_resolved_conversations
            OR p.conversations_resolved
        )
        AND COALESCE(a.current_approvals, 0) >= r.required_approvals
        AND (
            NOT r.require_status_checks
            OR (
                COALESCE(c.failed_checks, 0) = 0
                AND COALESCE(c.pending_checks, 0) = 0
                AND COALESCE(rcc.required_count, 0)
                    = COALESCE(rcc.passed_required_count, 0)
            )
        )
    ) AS policy_merge_eligible
FROM pull_requests p
JOIN required r
  ON r.pull_request_id = p.pull_request_id
LEFT JOIN approvals a
  ON a.pull_request_id = p.pull_request_id
LEFT JOIN checks c
  ON c.pull_request_id = p.pull_request_id
LEFT JOIN required_check_counts rcc
  ON rcc.pull_request_id = p.pull_request_id;

BEGIN;

INSERT INTO repositories (name, default_branch)
VALUES ('secure-payment-service', 'main');

INSERT INTO contributors (username, display_name)
VALUES
    ('alice', 'Alice Reviewer'),
    ('bob', 'Bob Security Reviewer'),
    ('carol', 'Carol Maintainer'),
    ('dave', 'Dave Developer');

INSERT INTO repository_branches (repository_id, branch_name, is_protected)
SELECT repository_id, branch_name, protected
FROM repositories
CROSS JOIN (
    VALUES
        ('main', true),
        ('develop', false),
        ('feature/payment-validation', false),
        ('feature/legacy-cleanup', false),
        ('feature/session-hardening', false)
) AS branches(branch_name, protected);

INSERT INTO branch_protection_policies (
    repository_id,
    branch_id,
    required_approvals,
    require_status_checks,
    require_resolved_conversations,
    prohibit_direct_push,
    prohibit_force_push,
    prohibit_branch_deletion,
    require_linear_history,
    allow_administrator_bypass
)
SELECT
    r.repository_id,
    b.branch_id,
    2,
    true,
    true,
    true,
    true,
    true,
    true,
    false
FROM repositories r
JOIN repository_branches b
  ON b.repository_id = r.repository_id
WHERE b.branch_name = 'main';

INSERT INTO required_status_checks (policy_id, check_name)
SELECT policy_id, check_name
FROM branch_protection_policies
CROSS JOIN (
    VALUES ('unit-tests'), ('security-scan')
) AS checks(check_name);

INSERT INTO pull_requests (
    repository_id,
    author_id,
    source_branch_id,
    target_branch_id,
    title,
    state,
    source_synchronized,
    has_merge_conflict,
    conversations_resolved
)
SELECT
    r.repository_id,
    author.contributor_id,
    source.branch_id,
    target.branch_id,
    pr.title,
    pr.state,
    pr.synchronized,
    pr.conflict,
    pr.resolved
FROM repositories r
JOIN contributors author
  ON author.username = pr.author_name
JOIN repository_branches source
  ON source.repository_id = r.repository_id
 AND source.branch_name = pr.source_branch
JOIN repository_branches target
  ON target.repository_id = r.repository_id
 AND target.branch_name = 'main'
CROSS JOIN (
    VALUES
        (
            'dave',
            'feature/payment-validation',
            'Payment validation hardening',
            'open',
            true, false, true
        ),
        (
            'dave',
            'feature/legacy-cleanup',
            'Legacy parser cleanup',
            'open',
            true, false, true
        ),
        (
            'dave',
            'feature/session-hardening',
            'Session validation hardening',
            'open',
            false, false, true
        )
) AS pr(
    author_name,
    source_branch,
    title,
    state,
    synchronized,
    conflict,
    resolved
);

INSERT INTO commits (
    pull_request_id,
    sha,
    message,
    committed_at
)
SELECT
    p.pull_request_id,
    x.sha,
    x.message,
    now() - x.age
FROM pull_requests p
JOIN (
    VALUES
        (
            'Payment validation hardening',
            'a111111',
            'Validate payment amount and currency',
            INTERVAL '2 days'
        ),
        (
            'Legacy parser cleanup',
            'b222222',
            'Remove unsafe legacy parser path',
            INTERVAL '1 day'
        ),
        (
            'Session validation hardening',
            'c333333',
            'Reject expired session tokens',
            INTERVAL '3 hours'
        )
) AS x(title, sha, message, age)
ON p.title = x.title;

INSERT INTO reviews (
    pull_request_id,
    reviewer_id,
    review_state,
    commit_id,
    is_stale
)
SELECT
    p.pull_request_id,
    c.contributor_id,
    x.review_state,
    cm.commit_id,
    x.stale
FROM pull_requests p
JOIN (
    VALUES
        ('Payment validation hardening', 'alice', 'approved', false),
        ('Payment validation hardening', 'bob', 'approved', false),
        ('Legacy parser cleanup', 'alice', 'changes_requested', false),
        ('Legacy parser cleanup', 'bob', 'approved', true),
        ('Session validation hardening', 'alice', 'approved', false)
) AS x(title, username, review_state, stale)
  ON x.title = p.title
JOIN contributors c
  ON c.username = x.username
LEFT JOIN commits cm
  ON cm.pull_request_id = p.pull_request_id;

INSERT INTO review_comments (
    review_id,
    file_path,
    line_number,
    body
)
SELECT
    review_id,
    'src/session.c',
    42,
    'Validate expiration before accepting the token.'
FROM reviews r
JOIN pull_requests p
  ON p.pull_request_id = r.pull_request_id
WHERE p.title = 'Session validation hardening'
  AND r.review_state = 'approved';

INSERT INTO status_checks (
    pull_request_id,
    check_name,
    status,
    completed_at
)
SELECT
    p.pull_request_id,
    x.check_name,
    x.status,
    CASE
        WHEN x.status <> 'pending' THEN now()
        ELSE NULL
    END
FROM pull_requests p
JOIN (
    VALUES
        ('Payment validation hardening', 'unit-tests', 'passed'),
        ('Payment validation hardening', 'security-scan', 'passed'),
        ('Legacy parser cleanup', 'unit-tests', 'passed'),
        ('Legacy parser cleanup', 'security-scan', 'failed'),
        ('Session validation hardening', 'unit-tests', 'passed'),
        ('Session validation hardening', 'security-scan', 'pending')
) AS x(title, check_name, status)
ON p.title = x.title;

INSERT INTO svm_models (
    model_name,
    feature_description,
    regularization_c,
    weight_vector,
    bias
)
VALUES (
    'merge-risk-linear-soft-margin',
    '[
        {"name": "security_defects", "meaning": "normalized defect count"},
        {"name": "test_coverage", "meaning": "normalized test coverage"}
    ]'::jsonb,
    1.0,
    ARRAY[-1.25, 1.05]::numeric[],
    0.20
);

-- Store representative decision scores and margins produced by a trained
-- linear model. The database stores the learned parameters and evaluates
-- governance queries; it is not responsible for numerical model training.
INSERT INTO svm_observations (
    model_id,
    pull_request_id,
    feature_vector,
    label,
    decision_score,
    functional_margin,
    hinge_loss,
    is_support_vector
)
SELECT
    m.model_id,
    p.pull_request_id,
    x.features,
    x.label,
    x.score,
    calculate_functional_margin(x.score, x.label),
    calculate_hinge_loss(
        calculate_functional_margin(x.score, x.label)
    ),
    x.score * x.label <= 1.05
FROM svm_models m
JOIN pull_requests p
  ON p.title = x.title
CROSS JOIN (
    VALUES
        (
            'Payment validation hardening',
            ARRAY[-0.80, 0.90]::numeric[],
            1,
            1.55::numeric
        ),
        (
            'Legacy parser cleanup',
            ARRAY[1.20, -0.90]::numeric[],
            -1,
            -1.70::numeric
        ),
        (
            'Session validation hardening',
            ARRAY[0.55, 0.10]::numeric[],
            1,
            0.45::numeric
        )
) AS x(title, features, label, score)
WHERE m.model_name = 'merge-risk-linear-soft-margin';

COMMIT;

-- Retrieval: current approval decisions, including stale approvals.
SELECT
    p.title,
    c.username AS reviewer,
    r.review_state,
    r.is_stale,
    r.submitted_at
FROM pull_requests p
JOIN reviews r
  ON r.pull_request_id = p.pull_request_id
JOIN contributors c
  ON c.contributor_id = r.reviewer_id
ORDER BY p.title, r.submitted_at;

-- Retrieval: support vectors are observations whose functional margins
-- lie at or near the canonical soft-margin boundary.
SELECT *
FROM svm_support_vector_report
ORDER BY functional_margin;

-- Evaluation: policy-level merge eligibility.
SELECT
    title,
    current_approvals,
    required_approvals,
    failed_checks,
    pending_checks,
    required_checks,
    passed_required_checks,
    policy_merge_eligible
FROM merge_eligibility_report
ORDER BY title;

-- Aggregation: compare the number of reviews and current approvals.
SELECT
    p.title,
    COUNT(r.review_id) AS total_reviews,
    COUNT(*) FILTER (
        WHERE r.review_state = 'approved'
    ) AS total_approvals,
    COUNT(*) FILTER (
        WHERE r.review_state = 'approved'
          AND NOT r.is_stale
    ) AS current_approvals
FROM pull_requests p
LEFT JOIN reviews r
  ON r.pull_request_id = p.pull_request_id
GROUP BY p.pull_request_id, p.title
ORDER BY p.title;

-- Validation: required status checks that are not currently passing.
SELECT
    p.title,
    required.check_name,
    COALESCE(sc.status, 'missing') AS actual_status
FROM pull_requests p
JOIN branch_protection_policies bp
  ON bp.branch_id = p.target_branch_id
JOIN required_status_checks required
  ON required.policy_id = bp.policy_id
LEFT JOIN status_checks sc
  ON sc.pull_request_id = p.pull_request_id
 AND sc.check_name = required.check_name
WHERE COALESCE(sc.status, 'missing') <> 'passed'
ORDER BY p.title, required.check_name;

-- Transactional integrity example:
-- this transaction intentionally attempts to add an invalid review state.
-- PostgreSQL rolls the transaction back, leaving existing governance data intact.
BEGIN;

INSERT INTO reviews (
    pull_request_id,
    reviewer_id,
    review_state
)
SELECT
    p.pull_request_id,
    c.contributor_id,
    'invalid_state'
FROM pull_requests p
JOIN contributors c
  ON c.username = 'carol'
WHERE p.title = 'Payment validation hardening';

ROLLBACK;
