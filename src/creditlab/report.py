"""report.md from the pipeline results: findings first, then the tables
behind them. Wording follows the numbers (a significant GBM gain, a drifting
calibration or a binding appetite limit each read differently)."""
from __future__ import annotations

from pathlib import Path


def pct(x, d=1):
    return "n.a." if x is None else f"{x * 100:.{d}f}%"


def num(x, d=3):
    return "n.a." if x is None else f"{x:.{d}f}"


def count(x):
    return "n.a." if x is None else f"{x:,.0f}"


def money_m(x):
    return "n.a." if x is None else f"${x / 1e6:,.2f}m"


def _table(headers, rows) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def markdown(r: dict) -> str:
    champ = r["champion"]["model"]
    m = r["models"]
    t = m[champ]["test"]
    diff = r["auc_difference_gbm_minus_scorecard"]
    grid = r["pricing"]["grid"]
    fun = r["funnel"]
    port = fun["portfolio"]
    lines = ["# Credit pricing lab: results", ""]
    if r["synthetic"]:
        lines += ["**SYNTHETIC DATA: invented loans for testing the pipeline. No result here describes Lending Club "
                  "or any real borrower.**", ""]
    split = r["data"]["split"]
    lines += [f"Data: `{Path(r['data']['raw_path']).name}`, snapshot {r['data']['data_end']}, {count(r['data']['rows_staged'])} loans "
              f"staged. Out-of-time test window: {split['test']['from']} to {split['test']['to']} "
              f"({split['test']['loans']:,} loans).", "", "## Findings", ""]
    gain = ("significant" if diff["low"] > 0 else "not significant")
    lines.append(f"- **Model.** Out of time, the scorecard reaches AUC {num(m['scorecard']['test']['auc'])} "
                 f"(Gini {num(m['scorecard']['test']['gini'])}, KS {num(m['scorecard']['test']['ks'])}) and gradient "
                 f"boosting {num(m['gbm']['test']['auc'])} (Gini {num(m['gbm']['test']['gini'])}). The GBM's AUC "
                 f"difference is {diff['difference']:+.4f} (95% bootstrap interval {diff['low']:+.4f} to "
                 f"{diff['high']:+.4f}): {gain}. Champion: {champ}. {r['champion']['reason']}.")
    drift = t["calibration_in_the_large"]
    lines.append(f"- **Calibration.** Recalibrated on the validation window, the champion predicts a mean PD of "
                 f"{pct(t['mean_pd'])} against an observed out-of-time default rate of {pct(t['default_rate'])} "
                 f"({drift * 100:+.1f} points). Score PSI from training to test is "
                 f"{num(m[champ]['score_psi_train_to_test'])} ({m[champ]['psi_label']}): the applicant mix "
                 f"{'barely moved' if m[champ]['score_psi_train_to_test'] < 0.1 else 'moved'}, so "
                 f"{'the gap is drift in default rates at a given score, which PSI cannot see' if abs(drift) > 0.01 and m[champ]['score_psi_train_to_test'] < 0.1 else 'monitor both measures'}.")
    under = [b for b, g in grid.items() if g.get("margin_at_observed_default") is not None
             and g["margin_at_observed_default"] < 0]
    mid = r["pricing"]["target_margins"][len(r["pricing"]["target_margins"]) // 2]
    if under:
        lines.append(f"- **Pricing back-test.** At the {pct(mid, 0)} target margin, band(s) {', '.join(under)} would "
                     "lose money at the default rates actually observed out of time: the APR grid inherits the "
                     "model's calibration gap. Price off the latest vintage, or add a margin of conservatism.")
    else:
        lines.append(f"- **Pricing back-test.** At the {pct(mid, 0)} target margin every band keeps a positive margin "
                     "at the default rates observed out of time.")
    lc_gap = {b: g["lc_margin_over_cost_and_el"] for b, g in grid.items()}
    thin = [b for b, v in lc_gap.items() if v is not None and v < 0]
    lines.append(f"- **Against Lending Club's prices.** After this lender's {pct(r['pricing']['cost_of_funds'])} cost "
                 f"of funds and {pct(r['pricing']['opex'])} operating cost, Lending Club's median rate leaves "
                 + (f"a negative margin in band(s) {', '.join(thin)}" if thin else "a positive margin in every band")
                 + ": a balance-sheet lender with these costs cannot match it there.")
    lines.append(f"- **Optimal policy.** Approve band(s) {''.join(fun['approved_bands']) or 'none'}; booked expected "
                 f"loss {pct(port['el_rate'])} of balance against a {pct(fun['assumptions']['max_portfolio_el'])} "
                 f"appetite, risk-adjusted margin {pct(port['margin'])}, expected profit {money_m(port['profit'])} a "
                 f"year on this applicant pool. The appetite constraint "
                 f"{'binds' if fun['lagrange_multiplier'] > 0 else 'does not bind'}.")
    stress = r.get("appetite_stress") or []
    if stress:
        base_bands = "".join(fun["approved_bands"])
        dropped = [x for x in stress if len(x["approved_bands"]) < len(base_bands)]
        binds = [x for x in stress if x["appetite_binds"]]
        def apr_move(x):
            start = stress[0]["avg_apr"]
            if start is None or x["avg_apr"] is None or abs(x["avg_apr"] - start) < 1e-4:
                return ""
            why = (" (under the adverse-selection assumption a lower price lowers the PD of those who accept)"
                   if x["avg_apr"] < start else "")
            return f"; the balance-weighted APR moves from {pct(start)} to {pct(x['avg_apr'])}{why}"
        if dropped:
            first = dropped[0]
            gone = "".join(b for b in base_bands if b not in first["approved_bands"])
            lines.append(f"- **Loss appetite.** Tightened from {pct(fun['assumptions']['max_portfolio_el'])} to "
                         f"{pct(first['appetite'])}, band(s) {gone} drop out and profit falls to "
                         f"{money_m(first['profit'])} a year{apr_move(first)}. The stress table shows each step.")
        else:
            lowest = stress[-1]
            lines.append(f"- **Loss appetite.** Down to {pct(lowest['appetite'])}, every band stays approved"
                         + (f"; from {pct(binds[0]['appetite'])} the constraint binds, and at {pct(lowest['appetite'])} "
                            f"profit is {money_m(lowest['profit'])}{apr_move(lowest)}." if binds else
                            ", and the constraint never binds in this range."))
    lgd = r["lgd"]
    lgd_rows = r.get("lgd_sensitivity") or []
    alt = next((x for x in lgd_rows if abs(x["lgd"] - lgd["assumption"]) < 1e-9), None)
    if alt is not None and abs(lgd.get("used", lgd["assumption"]) - lgd["assumption"]) > 1e-9:
        lines.append(f"- **LGD from the data.** Pricing uses the {pct(lgd['used'])} that charged-off training loans "
                     f"actually lost, not the {pct(lgd['assumption'], 0)} assumption; at the assumption, expected "
                     f"profit would be {money_m(alt['profit'])} instead of {money_m(port['profit'])}.")
    sens = r["price_sensitivity"]
    lo, hi = sens[0], sens[-1]
    lines.append(f"- **Price sensitivity is an assumption.** Between {lo['multiplier']}x and {hi['multiplier']}x the "
                 f"assumed sensitivity, the optimal balance-weighted APR moves from {pct(lo['avg_apr'])} to "
                 f"{pct(hi['avg_apr'])} and profit from {money_m(lo['profit'])} to {money_m(hi['profit'])}. A price "
                 "test on live applications is the only way to pin it down.")
    lines += ["", "## Model comparison (out-of-time test window)", ""]
    rows = []
    for name in ("scorecard", "gbm"):
        e = m[name]
        rows.append([name, num(e["train"]["auc"]), num(e["valid"]["auc"]), num(e["test"]["auc"]), num(e["test"]["gini"]),
                     num(e["test"]["ks"]), num(e["test"]["brier"], 4), pct(e["test"]["mean_pd"]),
                     pct(e["test"]["default_rate"]), num(e["score_psi_train_to_test"])])
    lines += _table(["Model", "AUC train", "AUC valid", "AUC test", "Gini test", "KS test", "Brier test",
                     "Mean PD test", "Default rate test", "Score PSI"], rows)
    lines += ["", "AUC, Gini and KS use the raw score; Brier and mean PD use the PD recalibrated on the validation "
                  "window.", "", "## Data and split", ""]
    lines += _table(["Window", "From", "To", "Loans", "Defaults", "Default rate"],
                    [[k, v["from"], v["to"], f"{v['loans']:,}", f"{v['defaults']:,}", pct(v["default_rate"], 2)]
                     for k, v in split.items()])
    iv = r["features"]["information_value"]
    lines += ["", "## Features", "", f"Scorecard features (information value at least the minimum): "
              f"{', '.join(r['features']['scorecard_selected'])}.", ""]
    lines += _table(["Feature", "IV", "PSI train to test"],
                    [[k, num(v), num(r["features"]["feature_psi_train_to_test"].get(k))] for k, v in iv.items()])
    emp = lgd["empirical_train"]
    used = lgd.get("used", lgd["assumption"])
    lines += ["", "## Expected loss and the APR grid", "",
              f"LGD {pct(used)} ({lgd.get('basis', 'assumption')}). Charged-off loans in the training window lost "
              + (f"{pct(emp['lgd'])} of the balance outstanding ({emp['loans']:,} loans) after recoveries net of "
                 "collection fees" if emp.get("lgd") is not None else "nothing measurable (no charged-off loans)")
              + f"; the configured assumption is {pct(lgd['assumption'], 0)}.", ""]
    margins = r["pricing"]["target_margins"]
    rows = []
    for b, g in grid.items():
        rows.append([b, f"{g['loans']:,}", pct(g["mean_pd"], 2), pct(g.get("observed_default_rate"), 2),
                     pct(g["el_rate"], 2)] + [pct(g[f"apr_margin_{x:.0%}"], 2) for x in margins]
                    + [pct(g.get("margin_at_observed_default"), 2), pct(g["lc_median_rate"], 2)])
    lines += _table(["Band", "Loans", "Mean PD", "Observed DR", "EL rate"] + [f"APR at {pct(x, 0)} margin"
                    for x in margins] + [f"Margin at observed DR ({pct(mid, 0)} target)", "Lending Club median rate"],
                    rows)
    lines += ["", f"APR = cost of funds {pct(r['pricing']['cost_of_funds'])} + operating cost "
                  f"{pct(r['pricing']['opex'])} + EL rate + target margin, capped at {pct(r['pricing']['apr_cap'], 0)}.",
              "", "## Funnel and optimal APR by band", ""]
    rows = []
    for b, g in fun["bands"].items():
        rows.append([b, "yes" if g["approved"] else "no", pct(g.get("apr"), 2), pct(g.get("market_rate"), 2),
                     pct(g.get("take_up"), 1), count(g.get("booked_loans")), money_m(g.get("booked_balance")),
                     money_m(g.get("profit")), pct(g.get("margin"), 2), pct(g.get("el_rate"), 2)])
    lines += _table(["Band", "Approved", "APR", "Market rate", "Take-up", "Booked loans", "Avg balance", "Profit",
                     "Margin", "EL rate"], rows)
    a = fun["assumptions"]
    lines += ["", f"Assumptions (not estimated from data): maximum take-up {pct(a['max_take_up'], 0)}; price "
                  f"sensitivity {a['price_sensitivity']} per percentage point above the market rate (logistic); "
                  f"adverse selection {pct(a['adverse_selection'], 0)} more PD per point above market; margin floor "
                  f"{pct(a['min_margin'], 0)}; expected-loss appetite {pct(a['max_portfolio_el'], 0)} of balance. "
                  "Market rate = the median rate Lending Club charged the band's borrowers.", "",
              "## Price-sensitivity sensitivity", ""]
    rows = [[s["multiplier"], num(s["price_sensitivity"], 3), s["approved_bands"], pct(s["avg_apr"], 2),
             money_m(s["booked_balance"]), pct(s["el_rate"], 2), money_m(s["profit"])] for s in sens]
    lines += _table(["Multiplier", "Sensitivity", "Approved", "Avg APR", "Avg balance", "EL rate", "Profit"], rows)
    if lgd_rows:
        lines += ["", "## LGD sensitivity", "", "The funnel re-optimised at each LGD; the APR grid above is not "
                  "recomputed.", ""]
        lines += _table(["LGD", "Approved", "Avg APR", "EL rate", "Profit"],
                        [[pct(x["lgd"], 0), x["approved_bands"], pct(x["avg_apr"], 2), pct(x["el_rate"], 2),
                          money_m(x["profit"])] for x in lgd_rows])
    if stress:
        lines += ["", "## Loss-appetite stress", ""]
        lines += _table(["Appetite", "Binds", "Approved", "Avg APR", "EL rate", "Avg balance", "Profit"],
                        [[pct(x["appetite"], 1), "yes" if x["appetite_binds"] else "no", x["approved_bands"] or "none",
                          pct(x["avg_apr"], 2), pct(x["el_rate"], 2), money_m(x["booked_balance"]),
                          money_m(x["profit"])] for x in stress])
    lines += ["", "## Files", ""] + [f"- `{Path(v).name}`" for v in r["files"].values()]
    return "\n".join(lines) + "\n"
