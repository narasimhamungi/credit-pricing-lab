-- Target: default within {horizon} months. A loan defaults inside the window
-- when it ended charged off (or 31-120 days late at the snapshot) and its last
-- payment came within the first {last_paid_month} months, so the first missed
-- payment fell inside the window. The file has no charge-off date; this is the
-- closest observable definition. Loans whose window had not closed by the
-- snapshot date are flagged and excluded downstream.
CREATE OR REPLACE VIEW labelled AS
SELECT
    l.*,
    date_diff('month', issue_d, last_pymnt_d)                            AS months_to_last_payment,
    CASE WHEN loan_status IN ({default_statuses})
              AND (last_pymnt_d IS NULL
                   OR date_diff('month', issue_d, last_pymnt_d) <= {last_paid_month})
         THEN 1 ELSE 0 END                                               AS default_12m,
    issue_d + INTERVAL {horizon} MONTH <= DATE '{data_end}'              AS window_complete
FROM loans l;
