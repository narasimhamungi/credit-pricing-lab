# credit-pricing-lab

Consumer-loan credit risk from origination data to price: a 12-month probability-of-default model (a weight-of-evidence scorecard against gradient boosting) validated out of time, expected loss, a risk-based APR grid, and an approval and take-up funnel that chooses each risk band's price to maximise profit within a credit-risk appetite.

## Problem

A lender pricing unsecured personal loans has to answer three questions in order: how likely is each applicant to default, what does that cost, and what price maximises profit once some approved applicants walk away and those who accept a high price are riskier than average. Most public credit-risk projects stop at an AUC. This one carries the PD through to an APR and an approval policy, and is explicit about which numbers come from data and which are assumptions.

## Why it matters

The model that ranks well is not automatically the model to price with: a PD that drifts out of calibration misprices every band even when the ranking holds. And the price that maximises margin per loan is rarely the price that maximises profit, because take-up falls and adverse selection rises as the rate goes up.

## What was built

| Step | What it does |
|---|---|
| SQL feature store (`src/creditlab/sql`, DuckDB) | Typed staging of the raw file, the 12-month default label with an incomplete-window cut, application-time features |
| Leakage guard (`data.py`) | Refuses any feature known only after origination, and Lending Club's own grade, rate and instalment |
| Out-of-time split | Train 2013-2015, calibrate 2016, test 2017 (configurable) |
| Scorecard (`woe.py`, `models.py`) | Weight-of-evidence binning and information value on the training window, logistic regression, points table (PDO 20, 600 at 50:1) |
| Gradient boosting | scikit-learn histogram GBM with native missing values and monotone constraints (FICO, DTI, inquiries, utilisation, loan-to-income, income) |
| Validation (`metrics.py`) | AUC, Gini, KS, Brier, log loss, reliability by decile, calibration-in-the-large, score and feature PSI, bootstrap interval on the AUC difference |
| Champion rule (`pipeline.py`) | The scorecard is kept unless the GBM's out-of-time AUC gain is significant |
| Expected loss and APR grid (`pricing.py`) | EL = PD x LGD x EAD; APR = cost of funds + opex + EL + target margin per band; back-test of the margin at the observed default rate; comparison with the rate Lending Club charged |
| Funnel and optimisation (`funnel.py`) | Approval and take-up against APR, adverse selection, profit-maximising APR per band and approval set under an APR cap, a margin floor and an expected-loss appetite; sensitivity to the price-response assumption and to LGD; a stress that tightens the loss appetite until bands drop out |

## Data and tools

- **Data:** Lending Club accepted loans 2007-2018Q4 (`accepted_2007_to_2018Q4.csv.gz`, the Kaggle mirror of Lending Club's loan-level files). The dataset page states a CC0 public-domain dedication (`docs/sources.md`). This repository still contains no loan-level data: only code, configuration and aggregate outputs.
- **Not used:** Home Credit's data set, released for a Kaggle competition and governed by that competition's rules.
- **Offline:** `creditlab run --synthetic` builds an invented file in the same layout (with the published file's quirks: `%` signs, `Mon-YYYY` dates, summary rows at the end) and runs everything on it; every output is marked SYNTHETIC.
- **Stack:** Python 3.11+, DuckDB, pandas, scikit-learn, matplotlib, pytest.

## Methodology

**Target.** Default within 12 months of issue: the loan ended charged off (or 31-120 days late at the snapshot) and its last payment came in the first 11 months, so the first missed payment fell inside the window. Lending Club's file has no charge-off date; this is the closest observable definition. Loans whose 12-month window had not closed by the file's snapshot date are excluded.

**Features.** Application-time only: amount, term, income, loan-to-income, DTI, FICO, inquiries, accounts, utilisation, revolving balance to income, delinquencies, public records, mortgages, credit history length, employment length, home ownership, verification, purpose, application type. Grade, sub-grade, rate and instalment are excluded: they are Lending Club's risk model and its price, so using them would copy its view and make the pricing circular. No protected attributes; state and ZIP code are left out to avoid geographic proxies.

**Models and calibration.** Both models are fitted on the training window. Their raw PDs are recalibrated with isotonic regression on the 2016 window, because default rates drift between vintages. Discrimination is measured on raw scores, calibration on recalibrated PDs, everything on the 2017 window the models never saw.

**Champion.** Gradient boosting replaces the scorecard only if the lower end of the 95% bootstrap interval on its out-of-time AUC gain is above zero. Otherwise the explainable model stays.

**Expected loss.** PD is the champion's calibrated 12-month PD. LGD is measured, not assumed: the share of the outstanding balance that charged-off loans in the training window lost after recoveries net of collection fees (`lgd_source: empirical`). The 85% assumption it replaces is kept as a comparison, and the funnel is re-optimised at LGDs of 80% to 95%. EAD is the average scheduled balance over the first year of the amortising loan, so per unit of balance the annual EL rate is PD x LGD.

**APR grid.** Seven bands on calibrated PD. APR = cost of funds + operating cost + EL rate + target margin (2%, 3%, 4%), capped at 36%. The back-test column shows the margin each band would earn at the default rate actually observed out of time.

**Funnel.** Take-up is a logistic function of the APR relative to the band's market rate (the median rate Lending Club charged those borrowers), and the PD of those who accept rises with the price (adverse selection). Both are assumptions: the file holds booked loans only, so take-up cannot be estimated from it. Each subset of bands is tried as the approval policy; within a subset the expected-loss appetite is priced with a Lagrange multiplier, so a band's APR trades its own profit against the room it takes under the limit. The most profitable feasible policy wins. The whole optimisation is repeated with the price sensitivity scaled from 0.5x to 2x.

## Key findings

Lending Club accepted loans, file snapshot 2019-03-01 (2,260,668 loans staged), split by origination date: train 2013-2015 (791,538 loans, 4.99% default within 12 months), validation 2016 (434,407, 6.16%) and an out-of-time test year, 2017 (443,579, 5.72%). The report with every table behind these numbers is `outputs/report.md`, and `outputs/metrics.json` holds each number.

- **Gradient boosting beats the scorecard out of time, and the difference is not noise.** Test AUC 0.672 (Gini 0.343, KS 0.246) against 0.655 (Gini 0.309, KS 0.220); the AUC gain is +0.0171 with a 95% bootstrap interval of +0.0153 to +0.0189. The GBM is the champion, with monotone constraints on FICO, DTI, inquiries, revolving utilisation, loan-to-income and income.
- **The champion is calibrated out of time.** Recalibrated on the validation year, it predicts a mean PD of 6.0% against an observed 5.7% in 2017; score PSI is 0.009 for the GBM and 0.030 for the scorecard. An independent validation in [mv-lab](https://github.com/narasimhamungi/mv-lab) finds no grade whose PD is too low; grade G's is conservative (17.6% predicted, 15.4% observed). The drivers are modest, as expected of Lending Club: FICO has the highest information value (0.097), then DTI (0.065), inquiries in the last six months (0.057), loan-to-income (0.052) and income verification (0.051). Revolving utilisation shows the largest feature drift (PSI 0.100).
- **LGD comes from the data, and it costs profit.** Charged-off loans in the training window (137,988) lost 89.5% of the balance outstanding after recoveries net of collection fees; pricing uses that, not the 85% assumption it replaces. At 85%, expected profit would be $24.7m instead of $22.6m; across LGDs of 80% to 95% it runs from $27.2m to $20.3m.
- **Cost-plus APRs run from 11.6% (band A) to 26.2% (band G) at a 3% target margin**, on a 5.5% cost of funds, 2.0% operating cost and expected loss. At the default rates observed in 2017 every band keeps a positive margin (2.93% to 4.96%); the model over-predicts PD in the two worst bands, so those margins come out above target.
- **Lending Club's own median rates would lose money for this lender in bands A, F and G** after those costs, and sit below the APR that earns a 2% margin in every band.
- **The optimal policy approves all seven bands**, with expected loss of 5.4% of balance against a 6.0% appetite that does not bind, a risk-adjusted margin of 3.1% and expected profit of $22.6m a year on the 2017 applicant pool. It is the price per band that does the work, from 12.0% (A) to 31.5% (G), where take-up falls to 0.5%.
- **Tightening the loss appetite shows the constraint at work.** At 5.0% it binds and is met by repricing alone ($22.2m); at 4.5% band G drops out ($20.9m); at 3.0% only A to E remain ($12.9m). The approved set is not nested: G returns at 4.0% and 3.5%, at lower prices. The model's adverse-selection assumption, under which a lower price lowers the PD of those who accept, is what lets a cheaper G fit inside a tighter appetite.
- **Profit is driven by an assumption the data cannot supply.** Scaling the assumed price sensitivity from 0.5x to 2.0x moves the average APR from 19.6% to 14.2% and profit from $45.3m to $11.7m. The file holds booked loans only, so take-up cannot be estimated from it; a price test on live applications is the only way to pin it down.

**Limits.** Take-up, price sensitivity, adverse selection, cost of funds, operating cost and the expected-loss appetite are assumptions in `config/settings.yml`. LGD is one portfolio-wide rate from the training window, not a model by segment. The default label approximates a 12-month default because the file has no charge-off date. The data ends in 2018 and reflects Lending Club's borrowers and rates, not the current rate environment.

On the synthetic file (`creditlab run --synthetic`) the pipeline does what the tests require of it: it ranks FICO as the strongest driver of the invented default model, keeps the scorecard because the GBM's out-of-time gain is not significant, and flags a calibration gap that score PSI cannot see (the applicant mix is stable, so PSI is near zero, while default rates at a given score have drifted upward).

## Insight demonstrated

- A model can hold its ranking out of time and still misprice every band: calibration needs its own monitoring, and PSI on scores is not that monitor.
- Choosing a complex model needs evidence of a real gain; a bootstrap interval on the AUC difference is the minimum.
- Price optimisation in credit is constrained optimisation: margin per loan, take-up, adverse selection and the portfolio's loss appetite pull in different directions.

## Employer takeaway

Relevant to credit risk, consumer lending, pricing and risk analytics roles: an end-to-end PD model built and validated the way a model-risk reviewer would expect (out-of-time test, calibration, stability, a challenger with a significance test), carried through to expected loss and pricing decisions, with assumptions separated from estimates.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
# put accepted_2007_to_2018Q4.csv.gz in data/raw/ (or set CREDITLAB_RAW_PATH)
creditlab features                                       # SQL features and the split summary
creditlab run                                            # models, validation, pricing, funnel -> outputs/
creditlab run --synthetic                                # the same on invented data -> outputs/synthetic/
creditlab run --export-scores data/exports/pd_scores.parquet  # also loan-level scores per window, for independent validation
pytest
```

## Repository layout

```
config/settings.yml        data path, target, split windows, features, model, pricing and funnel settings
src/creditlab/sql/         01_staging.sql, 02_label.sql, 03_features.sql (DuckDB)
src/creditlab/             data, synthetic, woe, models, metrics, pricing, funnel, charts, report, pipeline, cli
tests/                     label definition, leakage guard, split, metrics, WoE, models, pricing, optimiser, pipeline
outputs/                   report.md, metrics.json, CSV tables, charts
```

## What the model does not do

- The 12-month PD is used as an annual default rate for 36- and 60-month loans; there is no lifetime PD curve or prepayment model.
- LGD is one portfolio-wide rate from the training window, not a model by segment; EAD ignores prepayment.
- Take-up, price sensitivity and adverse selection are assumptions, not estimates. Rejected applications and declined offers are not in the data.
- Pricing is a one-year view on average balance; no capital charge or tax.
- Calibration uses the 2016 window. In production the latest complete vintage would be used, with a margin of conservatism while it seasons.
