"""Approval and take-up funnel, and profit-maximising APRs under constraints.

For each band and candidate APR r, with m the band's market rate (the median
rate Lending Club charged those borrowers):

  take-up      T(r)  = max_take_up / (1 + exp(s * (r - m) * 100))
  selected PD  pd(r) = pd * (1 + a * max(0, (r - m) * 100))      adverse selection
  balance      B(r)  = T(r) * sum(amount * EAD(r, term))          average first-year balance
  profit       P(r)  = B(r) * (r - cost of funds - opex) - T(r) * LGD * sum(amount * pd(r) * EAD(r, term))

s (price sensitivity), a (adverse selection) and max_take_up are
assumptions: the data holds booked loans only, so take-up cannot be
estimated from it. The approved bands and their APRs maximise portfolio
profit subject to the APR cap, a risk-adjusted margin floor per band and a
ceiling on the booked expected-loss rate (credit appetite).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from creditlab.pricing import ead_factor


@dataclass(frozen=True)
class FunnelParams:
    cost_of_funds: float
    opex: float
    lgd: float
    apr_cap: float
    max_take_up: float
    price_sensitivity: float
    adverse_selection: float
    min_margin: float
    max_portfolio_el: float
    apr_step: float

    @classmethod
    def from_config(cls, cfg: dict, sensitivity_multiplier: float = 1.0) -> "FunnelParams":
        p, f = cfg["pricing"], cfg["funnel"]
        return cls(p["cost_of_funds"], p["opex"], p["lgd"], p["apr_cap"], f["max_take_up"],
                   f["price_sensitivity"] * sensitivity_multiplier, f["adverse_selection"], f["min_margin"],
                   f["max_portfolio_el"], f["apr_step"])


def take_up(r, m, fp: FunnelParams):
    return fp.max_take_up / (1 + np.exp(fp.price_sensitivity * (np.asarray(r) - m) * 100))


def pd_uplift(r, m, fp: FunnelParams):
    return 1 + fp.adverse_selection * np.maximum(0.0, (np.asarray(r) - m) * 100)


def band_curve(b: pd.DataFrame, m: float, grid: np.ndarray, fp: FunnelParams) -> pd.DataFrame:
    """Economics of one band at every APR on the grid (per year, $)."""
    t = take_up(grid, m, fp)
    up = pd_uplift(grid, m, fp)
    bal = np.zeros_like(grid)
    loss = np.zeros_like(grid)
    for term, g in b.groupby("term_months"):
        e = ead_factor(grid, term)
        bal += g["loan_amnt"].sum() * e
        loss += (g["loan_amnt"] * g["pd"]).sum() * e
    booked_bal = t * bal
    el = t * up * fp.lgd * loss
    revenue = booked_bal * grid
    cost = booked_bal * (fp.cost_of_funds + fp.opex)
    profit = revenue - cost - el
    with np.errstate(divide="ignore", invalid="ignore"):
        margin = np.where(booked_bal > 0, profit / booked_bal, np.nan)
        el_rate = np.where(booked_bal > 0, el / booked_bal, np.nan)
    return pd.DataFrame({"apr": grid, "take_up": t, "booked_balance": booked_bal, "booked_loans": t * len(b),
                         "booked_amount": t * b["loan_amnt"].sum(), "revenue": revenue, "funding_and_opex": cost,
                         "expected_loss": el, "profit": profit, "margin": margin, "el_rate": el_rate})


def _options(curve: pd.DataFrame, fp: FunnelParams) -> pd.DataFrame:
    return curve[(curve["margin"] >= fp.min_margin) & (curve["apr"] <= fp.apr_cap)].reset_index(drop=True)


class _Arrays:
    """A band's feasible options as NumPy arrays, for fast repeated choices."""

    def __init__(self, o: pd.DataFrame, fp: FunnelParams):
        self.frame = o
        self.profit = o["profit"].to_numpy()
        self.el = o["expected_loss"].to_numpy()
        self.bal = o["booked_balance"].to_numpy()
        self.excess = self.el - fp.max_portfolio_el * self.bal

    def best(self, lam: float) -> int:
        return int(np.argmax(self.profit - lam * self.excess))


def _pick(arrs: dict[str, _Arrays], subset: tuple[str, ...], lam: float) -> dict[str, int]:
    """For each band in the subset, the option maximising profit - lam * (EL - appetite * balance)."""
    return {b: arrs[b].best(lam) for b in subset}


def _sum(arrs: dict[str, _Arrays], pick: dict[str, int]) -> tuple[float, float, float]:
    bal = sum(arrs[b].bal[i] for b, i in pick.items())
    el = sum(arrs[b].el[i] for b, i in pick.items())
    profit = sum(arrs[b].profit[i] for b, i in pick.items())
    return bal, el, profit


def _totals(choice: dict) -> dict:
    rows = [r for r in choice.values() if r is not None]
    bal = sum(r["booked_balance"] for r in rows)
    el = sum(r["expected_loss"] for r in rows)
    profit = sum(r["profit"] for r in rows)
    return {"booked_balance": bal, "expected_loss": el, "profit": profit,
            "el_rate": el / bal if bal else None, "margin": profit / bal if bal else None}


def _within(arrs: dict[str, _Arrays], subset: tuple[str, ...], fp: FunnelParams) -> tuple[dict, float] | None:
    """Best options for a fixed set of approved bands under the appetite
    limit: unconstrained if that fits, otherwise the smallest Lagrange
    multiplier that brings the booked EL rate within the limit (bisection)."""
    def rate(lam):
        bal, el, _ = _sum(arrs, _pick(arrs, subset, lam))
        return el / bal if bal > 0 else 0.0

    if rate(0.0) <= fp.max_portfolio_el:
        return _pick(arrs, subset, 0.0), 0.0
    hi = 1.0
    while rate(hi) > fp.max_portfolio_el:
        hi *= 2
        if hi > 1e6:
            return None
    lo = 0.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if rate(mid) > fp.max_portfolio_el else (lo, mid)
    return _pick(arrs, subset, hi), hi


def optimise(apps: pd.DataFrame, market: dict[str, float], bands: list[str], fp: FunnelParams) -> dict:
    """Maximise portfolio profit over which bands to approve and the APR of
    each, subject to the APR cap, the margin floor and the booked expected-
    loss appetite.

    Every subset of bands is tried (seven bands make 127 non-empty sets).
    Within a subset the appetite constraint is priced with a Lagrange
    multiplier found by bisection, so each band's APR trades its own profit
    against the room it takes up under the limit. The most profitable
    feasible subset wins."""
    grid = np.round(np.arange(fp.cost_of_funds + fp.opex, fp.apr_cap + 1e-9, fp.apr_step), 6)
    curves = {b: band_curve(apps[apps["band"] == b], market[b], grid, fp) for b in bands
              if (apps["band"] == b).any()}
    opts = {b: _options(c, fp) for b, c in curves.items()}
    usable = [b for b in curves if not opts[b].empty]
    arrs = {b: _Arrays(opts[b], fp) for b in usable}
    best_pick, best_lam, best_profit = {}, 0.0, 0.0
    for mask in range(1, 2 ** len(usable)):
        subset = tuple(b for i, b in enumerate(usable) if mask >> i & 1)
        found = _within(arrs, subset, fp)
        if found is None:
            continue
        pick, lam = found
        profit = _sum(arrs, pick)[2]
        if profit > best_profit + 1e-9:
            best_pick, best_lam, best_profit = pick, lam, profit
    best = {b: opts[b].loc[i].to_dict() for b, i in best_pick.items()}
    rows = []
    for b in bands:
        if b not in curves:
            continue
        sub = apps[apps["band"] == b]
        if b in best:
            r, approved = best[b], True
        else:   # report the band's own best price even when it is declined
            o = opts[b]
            r, approved = (o.loc[o["profit"].idxmax()].to_dict() if not o.empty else {}), False
        rows.append({**r, "band": b, "approved": approved, "applicants": len(sub), "market_rate": market[b],
                     "mean_pd": float(sub["pd"].mean())})
    table = pd.DataFrame(rows).set_index("band") if rows else pd.DataFrame()
    return {"table": table, "approved": [b for b in bands if b in best], "curves": curves,
            "multiplier": best_lam, "portfolio": _totals(best)}


def funnel_summary(apps: pd.DataFrame, result: dict) -> pd.DataFrame:
    """Applicants -> approved -> booked, by band and in total."""
    rows = []
    t = result["table"]
    for band, g in apps.groupby("band"):
        approved = band in result["approved"]
        booked = float(t.loc[band, "booked_loans"]) if approved and band in t.index else 0.0
        rows.append({"band": band, "applicants": len(g), "approved": len(g) if approved else 0,
                     "booked": booked, "apr": float(t.loc[band, "apr"]) if approved else None,
                     "take_up": float(t.loc[band, "take_up"]) if approved else None})
    df = pd.DataFrame(rows).set_index("band")
    df.loc["total"] = [df["applicants"].sum(), df["approved"].sum(), df["booked"].sum(), None, None]
    return df


def _row(res: dict, bands: list[str]) -> dict:
    t = res["table"]
    approved = t[t["approved"]] if not t.empty else t
    avg_apr = float((approved["apr"] * approved["booked_balance"]).sum() / approved["booked_balance"].sum()) \
        if len(approved) else None
    row = {"approved_bands": "".join(res["approved"]), "profit": res["portfolio"]["profit"],
           "booked_balance": res["portfolio"]["booked_balance"], "el_rate": res["portfolio"]["el_rate"],
           "avg_apr": avg_apr, "appetite_binds": bool(res["multiplier"] > 0)}
    for b in bands:
        row[f"apr_{b}"] = float(t.loc[b, "apr"]) if (not t.empty and b in t.index) else None
    return row


def sensitivity(apps: pd.DataFrame, market: dict[str, float], bands: list[str], cfg: dict) -> pd.DataFrame:
    """Re-optimise with the price sensitivity scaled by each multiplier."""
    rows = []
    for mult in cfg["funnel"]["sensitivity_multipliers"]:
        fp = FunnelParams.from_config(cfg, mult)
        rows.append({"multiplier": mult, "price_sensitivity": fp.price_sensitivity,
                     **_row(optimise(apps, market, bands, fp), bands)})
    return pd.DataFrame(rows)


def lgd_sensitivity(apps: pd.DataFrame, market: dict[str, float], bands: list[str], cfg: dict) -> pd.DataFrame:
    """Re-optimise the funnel at each LGD in pricing.lgd_sensitivity (the APR grid is not recomputed)."""
    base = FunnelParams.from_config(cfg)
    return pd.DataFrame([{"lgd": v, **_row(optimise(apps, market, bands, replace(base, lgd=v)), bands)}
                         for v in cfg["pricing"].get("lgd_sensitivity", [])])


def appetite_stress(apps: pd.DataFrame, market: dict[str, float], bands: list[str], cfg: dict) -> pd.DataFrame:
    """Re-optimise at tighter expected-loss appetites: which bands survive, at what price and profit."""
    base = FunnelParams.from_config(cfg)
    return pd.DataFrame([{"appetite": v, **_row(optimise(apps, market, bands, replace(base, max_portfolio_el=v)), bands)}
                         for v in cfg["funnel"].get("appetite_stress", [])])
