# Credit pricing lab: results

Data: `accepted_2007_to_2018Q4.csv.gz`, snapshot 2019-03-01, 2,260,668 loans staged. Out-of-time test window: 2017-01-01 to 2017-12-01 (443,579 loans).

## Findings

- **Model.** Out of time, the scorecard reaches AUC 0.655 (Gini 0.309, KS 0.220) and gradient boosting 0.672 (Gini 0.343). The GBM's AUC difference is +0.0171 (95% bootstrap interval +0.0153 to +0.0189): significant. Champion: gbm. GBM's out-of-time AUC gain is significant (95% interval above zero).
- **Calibration.** Recalibrated on the validation window, the champion predicts a mean PD of 6.0% against an observed out-of-time default rate of 5.7% (+0.2 points). Score PSI from training to test is 0.009 (stable): the applicant mix barely moved, so monitor both measures.
- **Pricing back-test.** At the 3% target margin every band keeps a positive margin at the default rates observed out of time.
- **Against Lending Club's prices.** After this lender's 5.5% cost of funds and 2.0% operating cost, Lending Club's median rate leaves a negative margin in band(s) A, F, G: a balance-sheet lender with these costs cannot match it there.
- **Optimal policy.** Approve band(s) ABCDEFG; booked expected loss 5.4% of balance against a 6.0% appetite, risk-adjusted margin 3.1%, expected profit $22.59m a year on this applicant pool. The appetite constraint does not bind.
- **Loss appetite.** Tightened from 6.0% to 4.5%, band(s) G drop out and profit falls to $20.89m a year; the balance-weighted APR moves from 16.1% to 15.2% (under the adverse-selection assumption a lower price lowers the PD of those who accept). The stress table shows each step.
- **LGD from the data.** Pricing uses the 89.5% that charged-off training loans actually lost, not the 85% assumption; at the assumption, expected profit would be $24.65m instead of $22.59m.
- **Price sensitivity is an assumption.** Between 0.5x and 2.0x the assumed sensitivity, the optimal balance-weighted APR moves from 19.6% to 14.2% and profit from $45.29m to $11.65m. A price test on live applications is the only way to pin it down.

## Model comparison (out-of-time test window)

| Model | AUC train | AUC valid | AUC test | Gini test | KS test | Brier test | Mean PD test | Default rate test | Score PSI |
|---|---|---|---|---|---|---|---|---|---|
| scorecard | 0.667 | 0.657 | 0.655 | 0.309 | 0.220 | 0.0531 | 5.9% | 5.7% | 0.030 |
| gbm | 0.703 | 0.676 | 0.672 | 0.343 | 0.246 | 0.0529 | 6.0% | 5.7% | 0.009 |

AUC, Gini and KS use the raw score; Brier and mean PD use the PD recalibrated on the validation window.

## Data and split

| Window | From | To | Loans | Defaults | Default rate |
|---|---|---|---|---|---|
| train | 2013-01-01 | 2015-12-01 | 791,538 | 39,512 | 4.99% |
| valid | 2016-01-01 | 2016-12-01 | 434,407 | 26,765 | 6.16% |
| test | 2017-01-01 | 2017-12-01 | 443,579 | 25,385 | 5.72% |

## Features

Scorecard features (information value at least the minimum): fico, dti, inq_last_6mths, loan_to_income, verification_status, term_months, annual_inc_log, home_ownership, credit_history_years, purpose_group, mort_acc.

| Feature | IV | PSI train to test |
|---|---|---|
| fico | 0.097 | 0.045 |
| dti | 0.065 | 0.003 |
| inq_last_6mths | 0.057 | 0.027 |
| loan_to_income | 0.052 | 0.045 |
| verification_status | 0.051 | n.a. |
| term_months | 0.040 | 0.005 |
| annual_inc_log | 0.040 | 0.010 |
| home_ownership | 0.037 | n.a. |
| credit_history_years | 0.035 | 0.005 |
| purpose_group | 0.034 | n.a. |
| mort_acc | 0.033 | 0.036 |
| emp_years | 0.016 | 0.021 |
| revol_util | 0.006 | 0.100 |
| revol_bal_to_income | 0.006 | 0.031 |
| total_acc | 0.002 | 0.045 |
| loan_amnt | 0.002 | 0.039 |
| pub_rec | 0.001 | 0.000 |
| pub_rec_bankruptcies | 0.001 | 0.001 |
| open_acc | 0.001 | 0.012 |
| delinq_2yrs | 0.000 | 0.000 |
| application_type | 0.000 | n.a. |

## Expected loss and the APR grid

LGD 89.5% (empirical (training window)). Charged-off loans in the training window lost 89.5% of the balance outstanding (137,988 loans) after recoveries net of collection fees; the configured assumption is 85%.

| Band | Loans | Mean PD | Observed DR | EL rate | APR at 2% margin | APR at 3% margin | APR at 4% margin | Margin at observed DR (3% target) | Lending Club median rate |
|---|---|---|---|---|---|---|---|---|---|
| A | 49,108 | 1.27% | 1.35% | 1.14% | 10.64% | 11.64% | 12.64% | 2.93% | 7.35% |
| B | 90,300 | 2.84% | 2.78% | 2.54% | 12.04% | 13.04% | 14.04% | 3.05% | 10.42% |
| C | 49,545 | 4.29% | 4.26% | 3.84% | 13.34% | 14.34% | 15.34% | 3.03% | 11.99% |
| D | 97,605 | 5.62% | 5.45% | 5.03% | 14.53% | 15.53% | 16.53% | 3.15% | 12.74% |
| E | 105,262 | 8.21% | 8.05% | 7.34% | 16.84% | 17.84% | 18.84% | 3.14% | 14.99% |
| F | 36,786 | 11.92% | 10.90% | 10.67% | 20.17% | 21.17% | 22.17% | 3.92% | 16.02% |
| G | 14,973 | 17.58% | 15.39% | 15.74% | 25.24% | 26.24% | 27.24% | 4.96% | 18.06% |

APR = cost of funds 5.5% + operating cost 2.0% + EL rate + target margin, capped at 36%.

## Funnel and optimal APR by band

| Band | Approved | APR | Market rate | Take-up | Booked loans | Avg balance | Profit | Margin | EL rate |
|---|---|---|---|---|---|---|---|---|---|
| A | yes | 12.00% | 7.35% | 9.9% | 4,837 | $60.01m | $1.92m | 3.19% | 1.31% |
| B | yes | 13.75% | 10.42% | 14.3% | 12,877 | $161.86m | $5.59m | 3.45% | 2.80% |
| C | yes | 15.00% | 11.99% | 15.5% | 7,686 | $98.01m | $3.25m | 3.31% | 4.19% |
| D | yes | 16.25% | 12.74% | 13.6% | 13,261 | $173.18m | $5.52m | 3.19% | 5.56% |
| E | yes | 18.50% | 14.99% | 13.6% | 14,302 | $195.24m | $5.59m | 2.86% | 8.14% |
| F | yes | 22.50% | 16.02% | 5.6% | 2,070 | $31.67m | $0.70m | 2.22% | 12.78% |
| G | yes | 31.50% | 18.06% | 0.5% | 81 | $1.50m | $0.02m | 1.56% | 22.44% |

Assumptions (not estimated from data): maximum take-up 60%; price sensitivity 0.35 per percentage point above the market rate (logistic); adverse selection 3% more PD per point above market; margin floor 1%; expected-loss appetite 6% of balance. Market rate = the median rate Lending Club charged the band's borrowers.

## Price-sensitivity sensitivity

| Multiplier | Sensitivity | Approved | Avg APR | Avg balance | EL rate | Profit |
|---|---|---|---|---|---|---|
| 0.5 | 0.175 | ABCDEFG | 19.62% | $739.41m | 5.99% | $45.29m |
| 0.75 | 0.262 | ABCDEFG | 17.44% | $724.09m | 5.77% | $30.15m |
| 1.0 | 0.350 | ABCDEFG | 16.08% | $721.47m | 5.45% | $22.59m |
| 1.5 | 0.525 | ABCDEFG | 14.82% | $715.50m | 5.19% | $15.22m |
| 2.0 | 0.700 | ABCDEFG | 14.17% | $699.88m | 5.01% | $11.65m |

## LGD sensitivity

The funnel re-optimised at each LGD; the APR grid above is not recomputed.

| LGD | Approved | Avg APR | EL rate | Profit |
|---|---|---|---|---|
| 80% | ABCDEFG | 15.86% | 5.08% | $27.15m |
| 85% | ABCDEFG | 16.05% | 5.29% | $24.65m |
| 90% | ABCDEFG | 16.08% | 5.48% | $22.37m |
| 95% | ABCDEFG | 16.30% | 5.70% | $20.30m |

## Loss-appetite stress

| Appetite | Binds | Approved | Avg APR | EL rate | Avg balance | Profit |
|---|---|---|---|---|---|---|
| 6.0% | no | ABCDEFG | 16.08% | 5.45% | $721.47m | $22.59m |
| 5.0% | yes | ABCDEFG | 15.59% | 4.97% | $710.35m | $22.15m |
| 4.5% | yes | ABCDEF | 15.16% | 4.50% | $660.05m | $20.89m |
| 4.0% | yes | ABCDEFG | 14.65% | 3.99% | $588.92m | $18.64m |
| 3.5% | yes | ABCDEFG | 14.06% | 3.50% | $518.83m | $15.91m |
| 3.0% | yes | ABCDE | 13.58% | 3.00% | $416.41m | $12.85m |

## Files

- `information_value.csv`
- `scorecard_points.csv`
- `reliability_test.csv`
- `pricing_grid.csv`
- `optimal_pricing.csv`
- `funnel.csv`
- `price_sensitivity.csv`
- `lgd_sensitivity.csv`
- `appetite_stress.csv`
- `reliability_out_of_time.png`
- `pricing_vs_market.png`
- `profit_vs_apr.png`
- `price_sensitivity.png`
