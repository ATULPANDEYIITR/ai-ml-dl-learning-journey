DROP SCHEMA IF EXISTS linear_regression_lab CASCADE;
CREATE SCHEMA linear_regression_lab;
SET search_path TO linear_regression_lab;

-- PostgreSQL implementation of a linear-regression learning model.
-- The relational domain represents repository development activity.
-- Regression is used to estimate Pull Request review duration.
-- Governance constraints remain separate from statistical prediction.

CREATE TABLE repositories (
    repository_id BIGSERIAL PRIMARY KEY,
    repository_name TEXT NOT NULL UNIQUE,
    default_branch TEXT NOT NULL DEFAULT 'main',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE branches (
    branch_id BIGSERIAL PRIMARY KEY,
    repository_id BIGINT NOT NULL
        REFERENCES repositories(repository_id)
        ON DELETE CASCADE,
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
    repository_id BIGINT NOT NULL
        REFERENCES repositories(repository_id)
        ON DELETE CASCADE,
    source_branch_id BIGINT NOT NULL
        REFERENCES branches(branch_id),
    target_branch_id BIGINT NOT NULL
        REFERENCES branches(branch_id),
    author_id BIGINT NOT NULL
        REFERENCES developers(developer_id),
    title TEXT NOT NULL,
    state TEXT NOT NULL
        CHECK (
            state IN (
                'draft',
                'open',
                'changes_requested',
                'approved',
                'merged',
                'closed'
            )
        ),
    changed_files INTEGER NOT NULL
        CHECK (changed_files >= 0),
    review_duration_hours NUMERIC(12, 4),
    branch_up_to_date BOOLEAN NOT NULL DEFAULT FALSE,
    merge_conflict BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    merged_at TIMESTAMPTZ,
    CHECK (
        source_branch_id <> target_branch_id
    ),
    CHECK (
        review_duration_hours IS NULL
        OR review_duration_hours >= 0
    )
);

CREATE TABLE commits (
    commit_id BIGSERIAL PRIMARY KEY,
    pull_request_id BIGINT NOT NULL
        REFERENCES pull_requests(pull_request_id)
        ON DELETE CASCADE,
    commit_hash CHAR(40) NOT NULL UNIQUE,
    committed_by BIGINT NOT NULL
        REFERENCES developers(developer_id),
    message TEXT NOT NULL,
    committed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE reviews (
    review_id BIGSERIAL PRIMARY KEY,
    pull_request_id BIGINT NOT NULL
        REFERENCES pull_requests(pull_request_id)
        ON DELETE CASCADE,
    reviewer_id BIGINT NOT NULL
        REFERENCES developers(developer_id),
    review_state TEXT NOT NULL
        CHECK (
            review_state IN (
                'commented',
                'approved',
                'changes_requested',
                'dismissed'
            )
        ),
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    dismissed_at TIMESTAMPTZ
);

CREATE TABLE review_comments (
    comment_id BIGSERIAL PRIMARY KEY,
    review_id BIGINT NOT NULL
        REFERENCES reviews(review_id)
        ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    line_number INTEGER
        CHECK (line_number IS NULL OR line_number > 0),
    comment_text TEXT NOT NULL,
    resolved BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE status_checks (
    status_check_id BIGSERIAL PRIMARY KEY,
    pull_request_id BIGINT NOT NULL
        REFERENCES pull_requests(pull_request_id)
        ON DELETE CASCADE,
    check_name TEXT NOT NULL,
    status TEXT NOT NULL
        CHECK (
            status IN (
                'pending',
                'passed',
                'failed'
            )
        ),
    completed_at TIMESTAMPTZ,
    UNIQUE (pull_request_id, check_name)
);

CREATE TABLE branch_protection_policies (
    policy_id BIGSERIAL PRIMARY KEY,
    branch_id BIGINT NOT NULL UNIQUE
        REFERENCES branches(branch_id)
        ON DELETE CASCADE,
    require_pull_request BOOLEAN NOT NULL DEFAULT TRUE,
    require_up_to_date_branch BOOLEAN NOT NULL DEFAULT TRUE,
    require_conversation_resolution BOOLEAN NOT NULL DEFAULT TRUE,
    restrict_direct_pushes BOOLEAN NOT NULL DEFAULT TRUE,
    prevent_force_push BOOLEAN NOT NULL DEFAULT TRUE,
    prevent_deletion BOOLEAN NOT NULL DEFAULT TRUE,
    require_linear_history BOOLEAN NOT NULL DEFAULT FALSE,
    required_approvals INTEGER NOT NULL DEFAULT 1
        CHECK (required_approvals >= 0)
);

CREATE TABLE required_status_checks (
    policy_id BIGINT NOT NULL
        REFERENCES branch_protection_policies(policy_id)
        ON DELETE CASCADE,
    check_name TEXT NOT NULL,
    PRIMARY KEY (policy_id, check_name)
);

CREATE INDEX idx_pull_requests_repository_target
    ON pull_requests(repository_id, target_branch_id, state);

CREATE INDEX idx_pull_requests_review_duration
    ON pull_requests(review_duration_hours)
    WHERE review_duration_hours IS NOT NULL;

CREATE INDEX idx_reviews_pull_request_state
    ON reviews(pull_request_id, review_state);

CREATE INDEX idx_review_comments_unresolved
    ON review_comments(review_id, resolved)
    WHERE resolved = FALSE;

CREATE INDEX idx_status_checks_pull_request_status
    ON status_checks(pull_request_id, status);

INSERT INTO repositories (
    repository_name,
    default_branch
)
VALUES
    ('payment-platform', 'main'),
    ('analytics-platform', 'main');

INSERT INTO developers (
    username,
    display_name
)
VALUES
    ('alice', 'Alice Reviewer'),
    ('bob', 'Bob Reviewer'),
    ('carol', 'Carol Developer'),
    ('david', 'David Reviewer'),
    ('erin', 'Erin Developer');

INSERT INTO branches (
    repository_id,
    branch_name,
    is_protected
)
SELECT
    repository_id,
    branch_name,
    branch_name = 'main'
FROM (
    SELECT
        repository_id,
        unnest(ARRAY['main', 'develop']) AS branch_name
    FROM repositories
) AS generated_branches;

INSERT INTO pull_requests (
    repository_id,
    source_branch_id,
    target_branch_id,
    author_id,
    title,
    state,
    changed_files,
    review_duration_hours,
    branch_up_to_date,
    merge_conflict
)
SELECT
    r.repository_id,
    source_branch.branch_id,
    target_branch.branch_id,
    d.developer_id,
    sample.title,
    sample.state,
    sample.changed_files,
    sample.review_duration_hours,
    sample.branch_up_to_date,
    sample.merge_conflict
FROM (
    VALUES
        (
            'payment-platform',
            'develop',
            'main',
            'carol',
            'Refactor payment retry flow',
            'merged',
            42,
            16.8::NUMERIC,
            TRUE,
            FALSE
        ),
        (
            'payment-platform',
            'develop',
            'main',
            'erin',
            'Add payment audit events',
            'open',
            27,
            NULL::NUMERIC,
            TRUE,
            FALSE
        ),
        (
            'analytics-platform',
            'develop',
            'main',
            'erin',
            'Optimize dashboard query',
            'changes_requested',
            36,
            NULL::NUMERIC,
            FALSE,
            TRUE
        ),
        (
            'analytics-platform',
            'develop',
            'main',
            'carol',
            'Add cohort retention report',
            'merged',
            18,
            8.2::NUMERIC,
            TRUE,
            FALSE
        )
) AS sample (
    repository_name,
    source_branch,
    target_branch,
    author_username,
    title,
    state,
    changed_files,
    review_duration_hours,
    branch_up_to_date,
    merge_conflict
)
JOIN repositories r
    ON r.repository_name = sample.repository_name
JOIN branches source_branch
    ON source_branch.repository_id = r.repository_id
   AND source_branch.branch_name = sample.source_branch
JOIN branches target_branch
    ON target_branch.repository_id = r.repository_id
   AND target_branch.branch_name = sample.target_branch
JOIN developers d
    ON d.username = sample.author_username;

INSERT INTO commits (
    pull_request_id,
    commit_hash,
    committed_by,
    message
)
SELECT
    pr.pull_request_id,
    repeat(
        substr(
            md5(pr.pull_request_id::TEXT || sequence_number::TEXT),
            1,
            40
        ),
        1
    )::CHAR(40),
    d.developer_id,
    CASE sequence_number
        WHEN 1 THEN 'Implement primary change'
        WHEN 2 THEN 'Address reviewer feedback'
        ELSE 'Refine validation and tests'
    END
FROM pull_requests pr
JOIN developers d
    ON d.username IN ('carol', 'erin')
CROSS JOIN generate_series(1, 3) AS generated(sequence_number)
WHERE (
    pr.pull_request_id % 2 = 1
    AND d.username = 'carol'
)
OR (
    pr.pull_request_id % 2 = 0
    AND d.username = 'erin'
);

INSERT INTO reviews (
    pull_request_id,
    reviewer_id,
    review_state
)
SELECT
    pr.pull_request_id,
    d.developer_id,
    review_data.review_state
FROM (
    VALUES
        (1::BIGINT, 'alice', 'approved'),
        (1::BIGINT, 'bob', 'approved'),
        (2::BIGINT, 'alice', 'commented'),
        (2::BIGINT, 'bob', 'approved'),
        (3::BIGINT, 'alice', 'changes_requested'),
        (4::BIGINT, 'david', 'approved'),
        (4::BIGINT, 'alice', 'approved')
) AS review_data (
    pull_request_id,
    reviewer_username,
    review_state
)
JOIN pull_requests pr
    ON pr.pull_request_id = review_data.pull_request_id
JOIN developers d
    ON d.username = review_data.reviewer_username;

INSERT INTO review_comments (
    review_id,
    file_path,
    line_number,
    comment_text,
    resolved
)
SELECT
    review_id,
    'src/payment/RetryService.java',
    84,
    'The retry limit should be validated before scheduling the next attempt.',
    CASE
        WHEN review_state = 'approved' THEN TRUE
        ELSE FALSE
    END
FROM reviews
WHERE review_state IN (
    'approved',
    'commented',
    'changes_requested'
);

INSERT INTO status_checks (
    pull_request_id,
    check_name,
    status,
    completed_at
)
SELECT
    pr.pull_request_id,
    check_data.check_name,
    check_data.status,
    CASE
        WHEN check_data.status = 'pending'
            THEN NULL
        ELSE CURRENT_TIMESTAMP
    END
FROM pull_requests pr
CROSS JOIN (
    VALUES
        ('unit-tests', 'passed'),
        ('security-scan', 'passed'),
        ('build', 'passed')
) AS check_data(check_name, status);

INSERT INTO branch_protection_policies (
    branch_id,
    require_pull_request,
    require_up_to_date_branch,
    require_conversation_resolution,
    restrict_direct_pushes,
    prevent_force_push,
    prevent_deletion,
    require_linear_history,
    required_approvals
)
SELECT
    branch_id,
    TRUE,
    TRUE,
    TRUE,
    TRUE,
    TRUE,
    TRUE,
    FALSE,
    2
FROM branches
WHERE is_protected = TRUE;

INSERT INTO required_status_checks (
    policy_id,
    check_name
)
SELECT
    policy.policy_id,
    required.check_name
FROM branch_protection_policies policy
CROSS JOIN (
    VALUES
        ('unit-tests'),
        ('security-scan'),
        ('build')
) AS required(check_name);

-- Simple regression:
-- y = review duration
-- x = changed files
--
-- The slope is:
-- covariance(x,y) / variance(x)
--
-- The intercept is:
-- mean(y) - slope * mean(x)
WITH observations AS (
    SELECT
        changed_files::NUMERIC AS x,
        review_duration_hours AS y
    FROM pull_requests
    WHERE review_duration_hours IS NOT NULL
),
statistics AS (
    SELECT
        AVG(x) AS x_mean,
        AVG(y) AS y_mean,
        SUM((x - AVG(x) OVER ()) *
            (y - AVG(y) OVER ()))
            AS numerator,
        SUM((x - AVG(x) OVER ()) *
            (x - AVG(x) OVER ()))
            AS denominator
    FROM observations
)
SELECT
    y_mean - (
        numerator / NULLIF(denominator, 0)
    ) * x_mean AS intercept,
    numerator / NULLIF(denominator, 0) AS coefficient_changed_files
FROM statistics;

-- Multiple-regression-oriented training dataset.
-- The application layer can use these rows to fit:
--
-- review_duration =
--     intercept
--     + b1 * changed_files
--     + b2 * reviewer_count
--     + b3 * unresolved_discussions
--
-- reviewer_count counts distinct active reviewers.
-- unresolved_discussions counts unresolved review comments.

CREATE VIEW regression_training_data AS
SELECT
    pr.pull_request_id,
    pr.changed_files,
    COUNT(DISTINCT r.reviewer_id) AS reviewer_count,
    COUNT(
        DISTINCT CASE
            WHEN rc.resolved = FALSE
                THEN rc.comment_id
        END
    ) AS unresolved_discussions,
    pr.review_duration_hours
FROM pull_requests pr
LEFT JOIN reviews r
    ON r.pull_request_id = pr.pull_request_id
LEFT JOIN review_comments rc
    ON rc.review_id = r.review_id
WHERE pr.review_duration_hours IS NOT NULL
GROUP BY
    pr.pull_request_id,
    pr.changed_files,
    pr.review_duration_hours;

SELECT *
FROM regression_training_data
ORDER BY pull_request_id;

-- Determine the active, non-dismissed approvals for every Pull Request.

CREATE VIEW active_review_summary AS
SELECT
    pr.pull_request_id,
    pr.title,
    COUNT(*) FILTER (
        WHERE r.review_state = 'approved'
    ) AS approval_count,
    COUNT(*) FILTER (
        WHERE r.review_state = 'changes_requested'
    ) AS changes_requested_count,
    COUNT(rc.comment_id) FILTER (
        WHERE rc.resolved = FALSE
    ) AS unresolved_comment_count
FROM pull_requests pr
LEFT JOIN reviews r
    ON r.pull_request_id = pr.pull_request_id
LEFT JOIN review_comments rc
    ON rc.review_id = r.review_id
WHERE
    r.review_id IS NULL
    OR r.review_state <> 'dismissed'
GROUP BY
    pr.pull_request_id,
    pr.title;

SELECT *
FROM active_review_summary
ORDER BY pull_request_id;

-- Required status checks are evaluated separately from reviews.

CREATE VIEW status_check_summary AS
SELECT
    pr.pull_request_id,
    COUNT(required.check_name) AS required_check_count,
    COUNT(required.check_name) FILTER (
        WHERE sc.status = 'passed'
    ) AS passed_required_check_count,
    COUNT(required.check_name) FILTER (
        WHERE sc.status <> 'passed'
        OR sc.status IS NULL
    ) AS failed_or_missing_required_check_count
FROM pull_requests pr
JOIN branches target_branch
    ON target_branch.branch_id = pr.target_branch_id
JOIN branch_protection_policies policy
    ON policy.branch_id = target_branch.branch_id
JOIN required_status_checks required
    ON required.policy_id = policy.policy_id
LEFT JOIN status_checks sc
    ON sc.pull_request_id = pr.pull_request_id
   AND sc.check_name = required.check_name
GROUP BY pr.pull_request_id;

SELECT *
FROM status_check_summary
ORDER BY pull_request_id;

-- Merge eligibility combines branch policy, review state,
-- synchronization, conflicts, conversation resolution, and checks.
--
-- This query intentionally does not treat a high regression prediction
-- as a governance blocker. Statistical prediction and policy enforcement
-- are separate responsibilities.

WITH approval_state AS (
    SELECT
        pr.pull_request_id,
        COUNT(*) FILTER (
            WHERE r.review_state = 'approved'
        ) AS approvals,
        COUNT(*) FILTER (
            WHERE r.review_state = 'changes_requested'
        ) AS changes_requested,
        COUNT(rc.comment_id) FILTER (
            WHERE rc.resolved = FALSE
        ) AS unresolved_comments
    FROM pull_requests pr
    LEFT JOIN reviews r
        ON r.pull_request_id = pr.pull_request_id
       AND r.review_state <> 'dismissed'
    LEFT JOIN review_comments rc
        ON rc.review_id = r.review_id
    GROUP BY pr.pull_request_id
),
check_state AS (
    SELECT
        pr.pull_request_id,
        COUNT(required.check_name) AS required_checks,
        COUNT(required.check_name) FILTER (
            WHERE sc.status = 'passed'
        ) AS passed_checks
    FROM pull_requests pr
    JOIN branches target_branch
        ON target_branch.branch_id = pr.target_branch_id
    JOIN branch_protection_policies policy
        ON policy.branch_id = target_branch.branch_id
    JOIN required_status_checks required
        ON required.policy_id = policy.policy_id
    LEFT JOIN status_checks sc
        ON sc.pull_request_id = pr.pull_request_id
       AND sc.check_name = required.check_name
    GROUP BY pr.pull_request_id
)
SELECT
    pr.pull_request_id,
    pr.title,
    pr.state,
    policy.required_approvals,
    approval_state.approvals,
    approval_state.changes_requested,
    approval_state.unresolved_comments,
    check_state.required_checks,
    check_state.passed_checks,
    (
        pr.state IN ('open', 'approved')
        AND pr.branch_up_to_date
        AND NOT pr.merge_conflict
        AND approval_state.approvals >= policy.required_approvals
        AND approval_state.changes_requested = 0
        AND approval_state.unresolved_comments = 0
        AND check_state.required_checks =
            check_state.passed_checks
    ) AS merge_eligible
FROM pull_requests pr
JOIN branches target_branch
    ON target_branch.branch_id = pr.target_branch_id
JOIN branch_protection_policies policy
    ON policy.branch_id = target_branch.branch_id
JOIN approval_state
    ON approval_state.pull_request_id = pr.pull_request_id
JOIN check_state
    ON check_state.pull_request_id = pr.pull_request_id
ORDER BY pr.pull_request_id;

-- Demonstrate a transaction that updates a review conversation and then
-- re-evaluates the data. PostgreSQL transactions ensure both statements
-- either become visible together or neither becomes committed.

BEGIN;

UPDATE review_comments
SET resolved = TRUE
WHERE comment_id = (
    SELECT MIN(comment_id)
    FROM review_comments
    WHERE resolved = FALSE
);

UPDATE pull_requests
SET branch_up_to_date = TRUE,
    merge_conflict = FALSE
WHERE pull_request_id = 3;

COMMIT;

-- Recalculate the governance state after the transaction.

WITH review_state AS (
    SELECT
        pr.pull_request_id,
        COUNT(*) FILTER (
            WHERE r.review_state = 'approved'
        ) AS approvals,
        COUNT(*) FILTER (
            WHERE r.review_state = 'changes_requested'
        ) AS changes_requested
    FROM pull_requests pr
    LEFT JOIN reviews r
        ON r.pull_request_id = pr.pull_request_id
       AND r.review_state <> 'dismissed'
    GROUP BY pr.pull_request_id
)
SELECT
    pr.pull_request_id,
    pr.title,
    pr.branch_up_to_date,
    pr.merge_conflict,
    review_state.approvals,
    review_state.changes_requested
FROM pull_requests pr
JOIN review_state
    ON review_state.pull_request_id =
       pr.pull_request_id
ORDER BY pr.pull_request_id;

-- The database-level constraints prevent malformed states such as:
-- negative changed-file counts, negative review duration, duplicate
-- repository branch names, duplicate status checks for one Pull Request,
-- and a Pull Request whose source and target branch are identical.
