-- Typed staging of the raw Lending Club file (read with every column as text).
-- Rates and utilisation may carry a '%' sign and dates are 'Mon-YYYY' in the
-- published file; both are parsed here. Footer rows with no loan amount (the
-- file ends with summary lines) are dropped.
CREATE OR REPLACE TABLE loans AS
SELECT
    id,
    TRY_CAST(loan_amnt AS DOUBLE)                                        AS loan_amnt,
    TRY_CAST(funded_amnt AS DOUBLE)                                      AS funded_amnt,
    TRY_CAST(regexp_extract(term, '(\d+)', 1) AS INTEGER)                AS term_months,
    TRY_CAST(replace(int_rate, '%', '') AS DOUBLE) / 100                 AS lc_rate,
    grade                                                                AS lc_grade,
    emp_length,
    upper(home_ownership)                                                AS home_ownership,
    verification_status,
    purpose,
    application_type,
    TRY_CAST(annual_inc AS DOUBLE)                                       AS annual_inc,
    TRY_CAST(dti AS DOUBLE)                                              AS dti,
    TRY_CAST(fico_range_low AS DOUBLE)                                   AS fico_low,
    TRY_CAST(fico_range_high AS DOUBLE)                                  AS fico_high,
    TRY_CAST(inq_last_6mths AS DOUBLE)                                   AS inq_last_6mths,
    TRY_CAST(open_acc AS DOUBLE)                                         AS open_acc,
    TRY_CAST(total_acc AS DOUBLE)                                        AS total_acc,
    TRY_CAST(pub_rec AS DOUBLE)                                          AS pub_rec,
    TRY_CAST(delinq_2yrs AS DOUBLE)                                      AS delinq_2yrs,
    TRY_CAST(pub_rec_bankruptcies AS DOUBLE)                             AS pub_rec_bankruptcies,
    TRY_CAST(mort_acc AS DOUBLE)                                         AS mort_acc,
    TRY_CAST(revol_bal AS DOUBLE)                                        AS revol_bal,
    TRY_CAST(replace(revol_util, '%', '') AS DOUBLE)                     AS revol_util,
    CAST(try_strptime(issue_d, '%b-%Y') AS DATE)                         AS issue_d,
    CAST(try_strptime(earliest_cr_line, '%b-%Y') AS DATE)                AS earliest_cr_line,
    CAST(try_strptime(last_pymnt_d, '%b-%Y') AS DATE)                    AS last_pymnt_d,
    loan_status,
    TRY_CAST(total_rec_prncp AS DOUBLE)                                  AS total_rec_prncp,
    TRY_CAST(recoveries AS DOUBLE)                                       AS recoveries,
    TRY_CAST(collection_recovery_fee AS DOUBLE)                          AS collection_recovery_fee
FROM raw
WHERE TRY_CAST(loan_amnt AS DOUBLE) IS NOT NULL
  AND try_strptime(issue_d, '%b-%Y') IS NOT NULL;
