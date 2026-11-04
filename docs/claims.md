# Claims ledger

Generated 2026-11-04 from the real run (Lending Club, test year 2017, 443,579 loans). Tags: Demonstrated / Inferred / Assumed / Unsupported.

| Claim | Tag | Evidence |
|---|---|---|
| Out-of-time PD validation with calibration, stability and a significance test for the challenger | Demonstrated on real data | `outputs/report.md`, `metrics.json` (GBM AUC 0.673 vs scorecard 0.655, bootstrap +0.0167 to +0.0200) |
| SQL feature pipeline with a 12-month default label and leakage guard | Demonstrated | three SQL files in the pipeline, 22 tests |
| Constrained, profit-maximising risk-based pricing with an approval funnel | Demonstrated; take-up economics, costs and the appetite are assumptions | `outputs/report.md` (profit $25.42m/yr, sensitivity $14.6m-$48.7m) |
| Lending Club AUC 0.673, 5.72% default rate, $25.4m profit | Demonstrated as model output; profit depends on assumed price sensitivity | `metrics.json` |
| "Elasticity estimated" | Unsupported — it is an assumption with a sensitivity sweep (0.5x-2x), not an estimate | README must say "assumed, swept 0.5x-2x," not "estimated" |
| Redistributable outputs | Unsupported until the dataset licence is confirmed — flagged as CC0 and recorded in `docs/sources.md` per 2 Nov upgrade, but confirm the Kaggle page's License field directly before this ledger is published | `docs/sources.md` |
| LGD 85% assumption | Flag: empirical training LGD is 89.5% (from the 2 Nov upgrade); if README still states 85% as the headline without the empirical figure alongside it as a sensitivity, correct | `metrics.json` lgd/empirical_train |

## Action (overclaim fixes)
1. Change "elasticity estimated" → "elasticity assumed (swept 0.5x–2x); not estimated from data."
2. Confirm `docs/sources.md` records the Kaggle licence field verbatim, not just "CC0 assumed."
3. If README leads with 85% LGD, add the 89.5% empirical figure as the primary number and 85% as the labelled sensitivity (per the 2 Nov upgrade already built).
