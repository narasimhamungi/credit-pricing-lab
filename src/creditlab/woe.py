"""Weight-of-evidence binning and information value, fitted on the training
window only.

WoE = ln(share of goods in the bin / share of bads in the bin), so a higher
WoE means lower risk. Numeric features get quantile bins (merged until each
holds a minimum share), missing values get their own bin, categories below
the minimum share are pooled into "other". Counts are smoothed by 0.5 so an
all-good or all-bad bin does not produce an infinite WoE.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MISSING = "missing"
OTHER = "other"


@dataclass
class FeatureBins:
    name: str
    kind: str                                   # "numeric" or "categorical"
    edges: list[float] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    woe: dict[str, float] = field(default_factory=dict)
    table: pd.DataFrame | None = None
    iv: float = 0.0

    def labels(self, x: pd.Series) -> pd.Series:
        if self.kind == "numeric":
            v = pd.to_numeric(x, errors="coerce")
            idx = np.searchsorted(self.edges, v.to_numpy(), side="right")
            lab = pd.Series([self._interval(i) for i in idx], index=x.index)
            return lab.where(v.notna(), MISSING)
        s = x.astype("object").where(x.notna(), MISSING).astype(str)
        return s.where(s.isin(self.categories) | (s == MISSING), OTHER)

    def _interval(self, i: int) -> str:
        lo = "-inf" if i == 0 else f"{self.edges[i - 1]:.4g}"
        hi = "inf" if i >= len(self.edges) else f"{self.edges[i]:.4g}"
        return f"[{lo}, {hi})"

    def transform(self, x: pd.Series) -> pd.Series:
        return self.labels(x).map(self.woe).fillna(0.0).astype(float)


def _table(labels: pd.Series, y: pd.Series) -> pd.DataFrame:
    t = pd.crosstab(labels, y).reindex(columns=[0, 1], fill_value=0)
    t.columns = ["goods", "bads"]
    g, b = t["goods"] + 0.5, t["bads"] + 0.5
    t["share"] = (t["goods"] + t["bads"]) / (t["goods"].sum() + t["bads"].sum())
    t["bad_rate"] = t["bads"] / (t["goods"] + t["bads"])
    t["woe"] = np.log((g / g.sum()) / (b / b.sum()))
    t["iv"] = ((g / g.sum()) - (b / b.sum())) * t["woe"]
    return t


def fit_numeric(name: str, x: pd.Series, y: pd.Series, bins: int, min_share: float) -> FeatureBins:
    v = pd.to_numeric(x, errors="coerce")
    present = v.dropna()
    qs = np.unique(np.quantile(present, np.linspace(0, 1, bins + 1)[1:-1])) if len(present) else np.array([])
    edges = list(qs)
    # merge bins smaller than the minimum share into a neighbour
    while edges:
        idx = np.searchsorted(edges, present.to_numpy(), side="right")
        shares = np.bincount(idx, minlength=len(edges) + 1) / max(len(v), 1)
        small = int(np.argmin(shares))
        if shares[small] >= min_share:
            break
        drop = small if small < len(edges) else small - 1
        edges.pop(drop)
    fb = FeatureBins(name, "numeric", edges=[float(e) for e in edges])
    fb.table = _table(fb.labels(x), y)
    fb.woe = fb.table["woe"].to_dict()
    fb.iv = float(fb.table["iv"].sum())
    return fb


def fit_categorical(name: str, x: pd.Series, y: pd.Series, min_share: float) -> FeatureBins:
    s = x.astype("object").where(x.notna(), MISSING).astype(str)
    shares = s.value_counts(normalize=True)
    cats = [c for c, sh in shares.items() if sh >= min_share and c != MISSING]
    fb = FeatureBins(name, "categorical", categories=cats)
    fb.table = _table(fb.labels(x), y)
    fb.woe = fb.table["woe"].to_dict()
    fb.iv = float(fb.table["iv"].sum())
    return fb


@dataclass
class WoeBinner:
    features: dict[str, FeatureBins]

    @classmethod
    def fit(cls, df: pd.DataFrame, y: pd.Series, numeric: list[str], categorical: list[str], bins: int = 10,
            min_share: float = 0.05) -> "WoeBinner":
        out = {}
        for f in numeric:
            out[f] = fit_numeric(f, df[f], y, bins, min_share)
        for f in categorical:
            out[f] = fit_categorical(f, df[f], y, min_share)
        return cls(out)

    def iv_table(self) -> pd.Series:
        return pd.Series({k: v.iv for k, v in self.features.items()}).sort_values(ascending=False)

    def transform(self, df: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
        names = names or list(self.features)
        return pd.DataFrame({n: self.features[n].transform(df[n]) for n in names}, index=df.index)
