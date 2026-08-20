CREATE VIEW vw_default_rate_by_grade AS
SELECT
    grade,
    COUNT(*) AS total_loans,
    SUM(CASE WHEN default_flag THEN 1 ELSE 0 END) AS defaults,
    ROUND(100.0 * SUM(CASE WHEN default_flag THEN 1 ELSE 0 END) / COUNT(*), 2) AS default_rate_pct,
    ROUND(AVG(int_rate), 2) AS avg_int_rate
FROM loans
GROUP BY grade
ORDER BY grade;


CREATE VIEW vw_vintage_analysis AS
SELECT
    issue_year,
    issue_quarter,
    COUNT(*) AS total_loans,
    SUM(CASE WHEN default_flag THEN 1 ELSE 0 END) AS defaults,
    ROUND(100.0 * SUM(CASE WHEN default_flag THEN 1 ELSE 0 END) / COUNT(*), 2) AS default_rate_pct
FROM loans
GROUP BY issue_year, issue_quarter
ORDER BY issue_year, issue_quarter;


CREATE VIEW vw_dti_income_segments AS
SELECT
    CASE
        WHEN dti < 10 THEN '0-10'
        WHEN dti < 20 THEN '10-20'
        WHEN dti < 30 THEN '20-30'
        WHEN dti < 40 THEN '30-40'
        ELSE '40+'
    END AS dti_bucket,
    CASE
        WHEN annual_inc < 40000 THEN 'Under 40K'
        WHEN annual_inc < 80000 THEN '40K-80K'
        WHEN annual_inc < 120000 THEN '80K-120K'
        ELSE '120K+'
    END AS income_bucket,
    COUNT(*) AS total_loans,
    ROUND(100.0 * SUM(CASE WHEN default_flag THEN 1 ELSE 0 END) / COUNT(*), 2) AS default_rate_pct
FROM loans
GROUP BY dti_bucket, income_bucket
ORDER BY dti_bucket, income_bucket;