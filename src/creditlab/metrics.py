"""Discrimination, calibration and stability measures."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def ks(y, score) -> float:
    """Maximum distance between the score distributions of bads and goods."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y = y[order]
    bads, goods = y.sum(), len(y) - y.sum()
    if bads == 0 or goods == 0:
        return float("nan")
    cum_b, cum_g = np.cumsum(y) / bads, np.cumsum(1 - y) / goods
    # evaluate only at the end of each run of tied scores
    last = np.r_[np.diff(score[order]) != 0, True]
    return float(np.max(np.abs(cum_b - cum_g)[last]))


def discrimination(y, score) -> dict:
    auc = roc_auc_score(y, score)
    return {"auc": float(auc), "gini": float(2 * auc - 1), "ks": ks(y, score)}


def calibration(y, pd_hat) -> dict:
    p = np.clip(np.asarray(pd_hat, dtype=float), 1e-6, 1 - 1e-6)
    return {"brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "mean_pd": float(p.mean()), "default_rate": float(np.mean(y)),
            "calibration_in_the_large": float(p.mean() - np.mean(y))}


def reliability(y, pd_hat, bins: int = 10) -> pd.DataFrame:
    """Observed default rate against mean predicted PD by PD decile."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(pd_hat, dtype=float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    out = df.groupby("bin").agg(loans=("y", "size"), mean_pd=("p", "mean"), default_rate=("y", "mean"))
    return out.reset_index(drop=True)


def psi(expected, actual, bins: int = 10) -> float:
    """Population stability index of `actual` against `expected`, on the
    expected distribution's quantile bins. Below 0.10 stable, 0.10-0.25 watch,
    above 0.25 a material shift."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e, a = e[~np.isnan(e)], a[~np.isnan(a)]
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)[1:-1]))
    ei = np.bincount(np.searchsorted(edges, e, side="right"), minlength=len(edges) + 1) / len(e)
    ai = np.bincount(np.searchsorted(edges, a, side="right"), minlength=len(edges) + 1) / len(a)
    ei, ai = np.clip(ei, 1e-4, None), np.clip(ai, 1e-4, None)
    return float(np.sum((ai - ei) * np.log(ai / ei)))


def psi_label(value: float) -> str:
    return "stable" if value < 0.10 else ("watch" if value < 0.25 else "shift")


def bootstrap_auc_difference(y, score_a, score_b, draws: int = 200, seed: int = 7) -> dict:
    """AUC(b) - AUC(a) with a percentile bootstrap interval, resampling loans."""
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(score_a, dtype=float), np.asarray(score_b, dtype=float)
    n = len(y)
    diffs = []
    for _ in range(draws):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    diffs = np.asarray(diffs)
    return {"difference": float(roc_auc_score(y, b) - roc_auc_score(y, a)), "low": float(np.quantile(diffs, 0.025)),
            "high": float(np.quantile(diffs, 0.975)), "draws": int(len(diffs))}
