-- ============================================================
-- Chess A/B Testing Project — SQL Query Library
-- Database: ab_test_dataset (loaded from ab_test_dataset.csv)
-- Compatible: PostgreSQL / SQLite / BigQuery (minor syntax diffs)
-- ============================================================


-- ─────────────────────────────────────────────────────────────
-- 0. TABLE CREATION  (SQLite / PostgreSQL)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ab_test_dataset (
    user_id             TEXT PRIMARY KEY,
    group_name          TEXT,           -- 'A' or 'B'
    strategy            TEXT,
    rating_before       INTEGER,
    rating_after        INTEGER,
    rating_improvement  INTEGER,
    improvement_pct     REAL,
    puzzles_attempted   INTEGER,
    puzzles_solved      INTEGER,
    avg_puzzle_rating   INTEGER,
    sessions            INTEGER,
    avg_session_min     REAL,
    games_played        INTEGER,
    wins                INTEGER,
    win_rate            REAL,
    days_active         INTEGER,
    returned_day7       INTEGER,        -- 1 = returned, 0 = churned
    returned_day30      INTEGER,
    engagement_score    REAL,
    skill_tier          TEXT
);


-- ─────────────────────────────────────────────────────────────
-- 1. OVERVIEW — sample size and baseline balance check
-- ─────────────────────────────────────────────────────────────
SELECT
    group_name                                        AS grp,
    strategy,
    COUNT(*)                                          AS n_users,
    ROUND(AVG(rating_before), 1)                      AS avg_baseline_rating,
    ROUND(STDDEV(rating_before), 1)                   AS std_baseline_rating,
    ROUND(MIN(rating_before), 0)                      AS min_rating,
    ROUND(MAX(rating_before), 0)                      AS max_rating
FROM ab_test_dataset
GROUP BY group_name, strategy
ORDER BY group_name;


-- ─────────────────────────────────────────────────────────────
-- 2. PRIMARY METRIC — average performance improvement
-- ─────────────────────────────────────────────────────────────
SELECT
    group_name,
    strategy,
    COUNT(*)                                          AS n_users,
    ROUND(AVG(rating_improvement), 2)                 AS avg_improvement_pts,
    ROUND(AVG(improvement_pct), 2)                    AS avg_improvement_pct,
    ROUND(STDDEV(rating_improvement), 2)              AS std_improvement,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP
          (ORDER BY rating_improvement), 1)           AS median_improvement,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP
          (ORDER BY rating_improvement), 1)           AS p25_improvement,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP
          (ORDER BY rating_improvement), 1)           AS p75_improvement
FROM ab_test_dataset
GROUP BY group_name, strategy
ORDER BY group_name;


-- ─────────────────────────────────────────────────────────────
-- 3. PRIMARY METRIC — retention rates (7-day and 30-day)
-- ─────────────────────────────────────────────────────────────
SELECT
    group_name,
    strategy,
    COUNT(*)                                               AS total_users,
    SUM(returned_day7)                                     AS returned_7d,
    SUM(returned_day30)                                    AS returned_30d,
    ROUND(100.0 * SUM(returned_day7)  / COUNT(*), 2)      AS retention_rate_7d_pct,
    ROUND(100.0 * SUM(returned_day30) / COUNT(*), 2)      AS retention_rate_30d_pct,
    -- Drop-off between 7d and 30d
    ROUND(
        100.0 * (SUM(returned_day7) - SUM(returned_day30))
              / NULLIF(SUM(returned_day7), 0), 2
    )                                                      AS dropoff_7_to_30d_pct
FROM ab_test_dataset
GROUP BY group_name, strategy
ORDER BY group_name;


-- ─────────────────────────────────────────────────────────────
-- 4. PRIMARY METRIC — engagement score
-- ─────────────────────────────────────────────────────────────
SELECT
    group_name,
    strategy,
    ROUND(AVG(engagement_score), 2)      AS avg_engagement,
    ROUND(AVG(sessions), 2)              AS avg_sessions,
    ROUND(AVG(avg_session_min), 2)       AS avg_session_min,
    ROUND(AVG(days_active), 2)           AS avg_days_active
FROM ab_test_dataset
GROUP BY group_name, strategy
ORDER BY group_name;


-- ─────────────────────────────────────────────────────────────
-- 5. SECONDARY METRIC — win rate
-- ─────────────────────────────────────────────────────────────
SELECT
    group_name,
    ROUND(AVG(win_rate) * 100, 2)         AS avg_win_rate_pct,
    ROUND(STDDEV(win_rate) * 100, 2)      AS std_win_rate_pct,
    ROUND(AVG(games_played), 1)           AS avg_games_played,
    SUM(wins)                             AS total_wins,
    SUM(games_played)                     AS total_games,
    ROUND(100.0 * SUM(wins) / SUM(games_played), 2) AS overall_win_rate_pct
FROM ab_test_dataset
GROUP BY group_name
ORDER BY group_name;


-- ─────────────────────────────────────────────────────────────
-- 6. SEGMENTED ANALYSIS — performance improvement by skill tier
-- ─────────────────────────────────────────────────────────────
SELECT
    skill_tier,
    group_name,
    COUNT(*)                                   AS n_users,
    ROUND(AVG(rating_improvement), 2)          AS avg_improvement,
    ROUND(AVG(improvement_pct), 2)             AS avg_improvement_pct,
    ROUND(AVG(retention_rate_7d_pct), 2)       AS retention_7d
FROM (
    SELECT *,
        ROUND(100.0 * returned_day7 / 1, 0)   AS retention_rate_7d_pct
    FROM ab_test_dataset
) t
GROUP BY skill_tier, group_name
ORDER BY
    CASE skill_tier
        WHEN 'Beginner'     THEN 1
        WHEN 'Intermediate' THEN 2
        WHEN 'Advanced'     THEN 3
        WHEN 'Expert'       THEN 4
        WHEN 'Master'       THEN 5
    END,
    group_name;


-- ─────────────────────────────────────────────────────────────
-- 7. COHORT COMPARISON — full side-by-side metrics table
-- ─────────────────────────────────────────────────────────────
WITH group_a AS (
    SELECT
        'A'                                     AS grp,
        COUNT(*)                                AS n,
        ROUND(AVG(rating_improvement), 2)       AS avg_improvement,
        ROUND(AVG(improvement_pct), 2)          AS avg_imp_pct,
        ROUND(100.0 * AVG(returned_day7), 2)    AS ret_7d,
        ROUND(100.0 * AVG(returned_day30), 2)   AS ret_30d,
        ROUND(AVG(engagement_score), 2)         AS engagement,
        ROUND(AVG(win_rate)*100, 2)             AS win_rate_pct,
        ROUND(AVG(avg_session_min), 2)          AS avg_sess_min
    FROM ab_test_dataset
    WHERE group_name = 'A'
),
group_b AS (
    SELECT
        'B'                                     AS grp,
        COUNT(*)                                AS n,
        ROUND(AVG(rating_improvement), 2)       AS avg_improvement,
        ROUND(AVG(improvement_pct), 2)          AS avg_imp_pct,
        ROUND(100.0 * AVG(returned_day7), 2)    AS ret_7d,
        ROUND(100.0 * AVG(returned_day30), 2)   AS ret_30d,
        ROUND(AVG(engagement_score), 2)         AS engagement,
        ROUND(AVG(win_rate)*100, 2)             AS win_rate_pct,
        ROUND(AVG(avg_session_min), 2)          AS avg_sess_min
    FROM ab_test_dataset
    WHERE group_name = 'B'
)
SELECT * FROM group_a
UNION ALL
SELECT * FROM group_b;


-- ─────────────────────────────────────────────────────────────
-- 8. UPLIFT CALCULATION — Group B lift over Group A
-- ─────────────────────────────────────────────────────────────
WITH averages AS (
    SELECT
        group_name,
        AVG(rating_improvement)       AS avg_improvement,
        AVG(improvement_pct)          AS avg_imp_pct,
        AVG(returned_day7)            AS ret_7d,
        AVG(returned_day30)           AS ret_30d,
        AVG(engagement_score)         AS engagement,
        AVG(win_rate)                 AS win_rate
    FROM ab_test_dataset
    GROUP BY group_name
),
pivoted AS (
    SELECT
        MAX(CASE WHEN group_name = 'A' THEN avg_improvement END)  AS a_improvement,
        MAX(CASE WHEN group_name = 'B' THEN avg_improvement END)  AS b_improvement,
        MAX(CASE WHEN group_name = 'A' THEN ret_7d END)           AS a_ret7d,
        MAX(CASE WHEN group_name = 'B' THEN ret_7d END)           AS b_ret7d,
        MAX(CASE WHEN group_name = 'A' THEN engagement END)       AS a_engagement,
        MAX(CASE WHEN group_name = 'B' THEN engagement END)       AS b_engagement
    FROM averages
)
SELECT
    ROUND(b_improvement - a_improvement, 2)              AS improvement_uplift_pts,
    ROUND(100.0*(b_improvement - a_improvement)
               / NULLIF(a_improvement, 0), 2)            AS improvement_uplift_pct,
    ROUND(100.0*(b_ret7d - a_ret7d), 2)                  AS retention_7d_uplift_pp,
    ROUND(100.0*(b_engagement - a_engagement)
               / NULLIF(a_engagement, 0), 2)             AS engagement_uplift_pct
FROM pivoted;


-- ─────────────────────────────────────────────────────────────
-- 9. TOP PERFORMERS — users with highest improvement in each group
-- ─────────────────────────────────────────────────────────────
SELECT
    user_id,
    group_name,
    strategy,
    skill_tier,
    rating_before,
    rating_after,
    rating_improvement,
    ROUND(improvement_pct, 2) AS improvement_pct,
    sessions,
    days_active
FROM (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY group_name
            ORDER BY rating_improvement DESC
        ) AS rn
    FROM ab_test_dataset
) ranked
WHERE rn <= 10
ORDER BY group_name, rn;


-- ─────────────────────────────────────────────────────────────
-- 10. USER-LEVEL AGGREGATION with engagement tier label
-- ─────────────────────────────────────────────────────────────
SELECT
    user_id,
    group_name,
    skill_tier,
    rating_before,
    rating_improvement,
    ROUND(improvement_pct, 2)              AS improvement_pct,
    sessions,
    ROUND(avg_session_min, 1)              AS avg_session_min,
    days_active,
    ROUND(engagement_score, 2)             AS engagement_score,
    CASE
        WHEN engagement_score >= 20 THEN 'High'
        WHEN engagement_score >= 12 THEN 'Medium'
        ELSE 'Low'
    END                                    AS engagement_tier,
    ROUND(win_rate * 100, 1)               AS win_rate_pct,
    returned_day7,
    returned_day30
FROM ab_test_dataset
ORDER BY group_name, rating_improvement DESC;
