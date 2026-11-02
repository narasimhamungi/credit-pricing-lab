"""Risk bands, expected loss and the APR grid.

EL = PD x LGD x EAD. PD is the model's calibrated 12-month PD; LGD is an
assumption (the empirical recovery rate on charged-off loans in the training
window is reported beside it); EAD is the average scheduled balance over the
first year of an amortising loan, so per unit of balance the annual EL rate
is PD x LGD.

APR = cost of funds + operating cost + EL rate + target margin, per band,
capped at the APR ceiling. The grid shows each target margin, the margin the
band would actually earn at its observed default rate, and the rate Lending
Club charged the same borrowers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def assign_bands(pd_hat, upper: list[float], names: list[str]) -> np.ndarray:
    if len(names) != len(upper) + 1:
        raise ValueError("need one more band name than band boundaries")
    return np.asarray(names, dtype=object)[np.searchsorted(np.asarray(upper), np.asarray(pd_hat), side="right")]


def ead_factor(apr, term_months, horizon: int = 12):
    """Average scheduled balance over the first `horizon` months, as a share
    of the amount lent, for a level-payment loan. Vectorised over apr/term."""
    apr = np.asarray(apr, dtype=float)
    n = np.asarray(term_months, dtype=float)
    i = apr / 12.0
    k = np.arange(horizon).reshape((-1,) + (1,) * max(np.ndim(apr), np.ndim(n)))
    with np.errstate(divide="ignore", invalid="ignore"):
        growth_n = (1 + i) ** n
        bal = np.where(i > 0, (growth_n - (1 + i) ** k) / (growth_n - 1), 1 - k / n)
    return bal.mean(axis=0)


def empirical_lgd(df: pd.DataFrame) -> dict:
    """Amount-weighted loss on charged-off loans: 1 - net recoveries / balance
    outstanding at charge-off. Recoveries still in progress at the snapshot
    make this an upper bound for recent vintages."""
    co = df[df["loan_status"].isin(["Charged Off", "Does not meet the credit policy. Status:Charged Off"])]
    if co.empty:
        return {"loans": 0, "lgd": None}
    ead = (co["funded_amnt"] - co["total_rec_prncp"]).clip(lower=0)
    net = (co["recoveries"].fillna(0) - co["collection_recovery_fee"].fillna(0)).clip(lower=0)
    total = ead.sum()
    return {"loans": int(len(co)), "lgd": float(1 - net.sum() / total) if total > 0 else None,
            "exposure": float(total)}


def band_table(apps: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """apps: one row per applicant with pd, band, default_12m (observed, may be
    absent), lc_rate, loan_amnt, term_months."""
    p = cfg["pricing"]
    cost = p["cost_of_funds"] + p["opex"]
    rows = []
    for band in p["band_names"]:
        b = apps[apps["band"] == band]
        if b.empty:
            continue
        mean_pd = float(b["pd"].mean())
        el = mean_pd * p["lgd"]
        row = {"band": band, "loans": len(b), "share": len(b) / len(apps), "mean_pd": mean_pd,
               "el_rate": el, "lc_median_rate": float(b["lc_rate"].median()) if "lc_rate" in b else np.nan}
        row["breakeven_apr"] = cost + el
        if "default_12m" in b:
            dr = float(b["default_12m"].mean())
            row["observed_default_rate"] = dr
            row["observed_el_rate"] = dr * p["lgd"]
            row["breakeven_apr_observed"] = cost + dr * p["lgd"]
        for m in p["target_margins"]:
            row[f"apr_margin_{m:.0%}"] = min(cost + el + m, p["apr_cap"])
        base_m = p["target_margins"][len(p["target_margins"]) // 2]
        apr = min(cost + el + base_m, p["apr_cap"])
        row["cap_binds"] = cost + el + base_m > p["apr_cap"]
        if "observed_default_rate" in row:
            row["margin_at_observed_default"] = apr - cost - row["observed_el_rate"]
        row["lc_margin_over_cost_and_el"] = row["lc_median_rate"] - cost - el
        rows.append(row)
    return pd.DataFrame(rows).set_index("band")
