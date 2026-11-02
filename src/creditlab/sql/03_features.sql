-- Application-time features. Nothing here is known only after origination:
-- payments, recoveries and Lending Club's grade, rate and instalment are kept
-- out of the feature list (some are carried for evaluation and LGD only).
CREATE OR REPLACE VIEW features AS
SELECT
    id, issue_d, default_12m, window_complete, loan_status, months_to_last_payment,
    -- carried for evaluation, pricing comparison and LGD; never used as features
    lc_rate, lc_grade, funded_amnt, total_rec_prncp, recoveries, collection_recovery_fee,
    -- features
    loan_amnt,
    term_months,
    ln(greatest(annual_inc, 1))                                          AS annual_inc_log,
    loan_amnt / nullif(annual_inc, 0)                                    AS loan_to_income,
    CASE WHEN dti < 0 OR dti > 100 THEN NULL ELSE dti END                AS dti,
    (fico_low + fico_high) / 2                                           AS fico,
    inq_last_6mths,
    open_acc,
    total_acc,
    revol_util,
    revol_bal / nullif(annual_inc, 0)                                    AS revol_bal_to_income,
    delinq_2yrs,
    pub_rec,
    pub_rec_bankruptcies,
    mort_acc,
    date_diff('month', earliest_cr_line, issue_d) / 12.0                 AS credit_history_years,
    CASE WHEN emp_length LIKE '10+%' THEN 10.0
         WHEN emp_length LIKE '<%' THEN 0.0
         ELSE TRY_CAST(regexp_extract(emp_length, '(\d+)', 1) AS DOUBLE) END AS emp_years,
    CASE WHEN home_ownership IN ('MORTGAGE', 'RENT', 'OWN') THEN home_ownership ELSE 'OTHER' END
                                                                         AS home_ownership,
    coalesce(verification_status, 'Unknown')                             AS verification_status,
    CASE WHEN purpose IN ('debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase',
                          'small_business', 'medical', 'car')
         THEN purpose ELSE 'other' END                                   AS purpose_group,
    coalesce(application_type, 'Individual')                             AS application_type
FROM labelled;
