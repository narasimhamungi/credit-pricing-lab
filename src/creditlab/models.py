"""Two PD models on the same application-time features.

Scorecard: weight-of-evidence bins and logistic regression, the standard
credit scorecard, converted to points (PDO, base score at base odds).
Gradient boosting: scikit-learn's histogram GBM with native missing values,
categorical support and monotone constraints on the features where the
direction of risk is not in doubt.

Both are fitted on the training window. Each model's PDs are then
recalibrated with isotonic regression on the validation window, because the
default rate drifts over time; discrimination is measured on the raw score,
calibration on the recalibrated PD, and everything on the out-of-time test
window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from creditlab.woe import WoeBinner


@dataclass
class Scorecard:
    binner: WoeBinner
    selected: list[str]
    lr: LogisticRegression
    pdo: float
    base_score: float
    base_odds: float

    def raw_pd(self, df: pd.DataFrame) -> np.ndarray:
        return self.lr.predict_proba(self.binner.transform(df, self.selected))[:, 1]

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        p = np.clip(self.raw_pd(df), 1e-9, 1 - 1e-9)
        return self.offset + self.factor * np.log((1 - p) / p)

    def points_table(self) -> pd.DataFrame:
        """Points per bin: score = sum of points across features."""
        n = len(self.selected)
        rows = []
        for coef, name in zip(self.lr.coef_[0], self.selected):
            fb = self.binner.features[name]
            for label, row in fb.table.iterrows():
                pts = self.offset / n - self.factor * (coef * row["woe"] + self.lr.intercept_[0] / n)
                rows.append({"feature": name, "bin": label, "loans": int(row["goods"] + row["bads"]),
                             "bad_rate": row["bad_rate"], "woe": row["woe"], "points": pts})
        return pd.DataFrame(rows)


def fit_scorecard(train: pd.DataFrame, cfg: dict) -> Scorecard:
    w, f = cfg["models"]["woe"], cfg["features"]
    y = train["default_12m"]
    binner = WoeBinner.fit(train, y, f["numeric"], f["categorical"], w["bins"], w["min_bin_share"])
    iv = binner.iv_table()
    selected = [k for k, v in iv.items() if v >= w["iv_min"]]
    if not selected:
        raise ValueError("no feature reaches the minimum information value")
    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(binner.transform(train, selected), y)
    return Scorecard(binner, selected, lr, w["pdo"], w["base_score"], w["base_odds"])


@dataclass
class Gbm:
    model: HistGradientBoostingClassifier
    numeric: list[str]
    categorical: list[str]
    categories: dict[str, list[str]]

    def frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[self.numeric].apply(pd.to_numeric, errors="coerce").copy()
        for c in self.categorical:
            out[c] = pd.Categorical(df[c].astype(str), categories=self.categories[c])
        return out

    def raw_pd(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.frame(df))[:, 1]


def fit_gbm(train: pd.DataFrame, cfg: dict) -> Gbm:
    g, f = cfg["models"]["gbm"], cfg["features"]
    numeric, categorical = list(f["numeric"]), list(f["categorical"])
    categories = {c: sorted(train[c].astype(str).unique().tolist()) for c in categorical}
    cols = numeric + categorical
    mono = [int(g.get("monotonic", {}).get(c, 0)) for c in cols]
    model = HistGradientBoostingClassifier(
        max_iter=g["max_iter"], learning_rate=g["learning_rate"], max_leaf_nodes=g["max_leaf_nodes"],
        min_samples_leaf=g["min_samples_leaf"], l2_regularization=g["l2_regularization"],
        early_stopping=g["early_stopping"], validation_fraction=0.15, categorical_features="from_dtype",
        monotonic_cst=mono, random_state=cfg["models"]["random_state"])
    gb = Gbm(model, numeric, categorical, categories)
    model.fit(gb.frame(train), train["default_12m"])
    return gb


@dataclass
class Calibrated:
    """A model plus an isotonic map from its raw PD to a recalibrated PD."""
    name: str
    model: Scorecard | Gbm
    iso: IsotonicRegression

    def raw(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.raw_pd(df)

    def pd(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.iso.predict(self.raw(df)), 1e-5, 1 - 1e-5)


def calibrate(name: str, model, valid: pd.DataFrame) -> Calibrated:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(model.raw_pd(valid), valid["default_12m"])
    return Calibrated(name, model, iso)
