import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from creditlab import metrics, models, woe


def test_ks_matches_scipy():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 4000)
    score = rng.normal(0, 1, 4000) + 0.8 * y
    expected = ks_2samp(score[y == 1], score[y == 0]).statistic
    assert metrics.ks(y, score) == pytest.approx(expected, abs=1e-12)
    assert metrics.ks([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_gini_and_psi():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 2000)
    d = metrics.discrimination(y, rng.random(2000) + y)
    assert d["gini"] == pytest.approx(2 * d["auc"] - 1)
    x = rng.normal(0, 1, 20000)
    assert metrics.psi(x, rng.normal(0, 1, 20000)) < 0.01
    assert metrics.psi(x, rng.normal(1.0, 1, 20000)) > 0.25
    assert metrics.psi_label(0.05) == "stable" and metrics.psi_label(0.3) == "shift"


def test_reliability_and_bootstrap():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.01, 0.3, 5000)
    y = (rng.random(5000) < p).astype(int)
    rel = metrics.reliability(y, p)
    assert len(rel) == 10 and rel["loans"].sum() == 5000
    assert np.corrcoef(rel["mean_pd"], rel["default_rate"])[0, 1] > 0.9
    same = metrics.bootstrap_auc_difference(y, p, p, draws=50)
    assert same["difference"] == 0 and same["low"] == 0 and same["high"] == 0


def test_woe_numeric_and_categorical():
    rng = np.random.default_rng(4)
    x = pd.Series(rng.uniform(0, 100, 10000))
    y = pd.Series((rng.random(10000) < 0.02 + 0.002 * x).astype(int))
    fb = woe.fit_numeric("x", x, y, bins=5, min_share=0.05)
    w = fb.table["woe"].to_numpy()
    assert len(fb.edges) == 4 and (np.diff(w) < 0).all()      # higher x, more bads, lower WoE
    assert fb.iv > 0.1
    missing = fb.transform(pd.Series([np.nan]))
    assert missing.iloc[0] == 0.0                             # no missing bin seen in training
    cat = pd.Series(rng.choice(["a", "b", "rare"], 10000, p=[0.6, 0.39, 0.01]))
    yc = pd.Series((rng.random(10000) < np.where(cat == "a", 0.05, 0.15)).astype(int))
    fc = woe.fit_categorical("c", cat, yc, min_share=0.05)
    assert fc.categories == ["a", "b"]
    assert set(fc.labels(pd.Series(["rare", "new", None]))) == {"other", "missing"}


def test_models_on_synthetic_data(parts, cfg):
    sc = models.fit_scorecard(parts.train, cfg)
    assert sc.selected[0] == "fico"                            # strongest driver in the generating model
    gb = models.fit_gbm(parts.train, cfg)
    for m in (sc, gb):
        auc = metrics.discrimination(parts.test["default_12m"], m.raw_pd(parts.test))["auc"]
        assert auc > 0.68
    cal = models.calibrate("scorecard", sc, parts.valid)
    before = metrics.calibration(parts.valid["default_12m"], sc.raw_pd(parts.valid))["brier"]
    after = metrics.calibration(parts.valid["default_12m"], cal.pd(parts.valid))["brier"]
    assert after <= before + 1e-12


def test_scorecard_points_add_up_to_the_score(parts, cfg):
    sc = models.fit_scorecard(parts.train, cfg)
    pts = sc.points_table()
    row = parts.test.iloc[[0]]
    total = 0.0
    for f in sc.selected:
        label = sc.binner.features[f].labels(row[f]).iloc[0]
        total += pts[(pts["feature"] == f) & (pts["bin"] == label)]["points"].iloc[0]
    assert total == pytest.approx(sc.score(row)[0], rel=1e-9)


def test_gbm_respects_monotone_constraints(parts, cfg):
    gb = models.fit_gbm(parts.train, cfg)
    row = parts.test.iloc[[0]].copy()
    probs = []
    for fico in (660, 700, 740, 780, 820):
        row["fico"] = fico
        probs.append(gb.raw_pd(row)[0])
    assert all(a >= b - 1e-12 for a, b in zip(probs, probs[1:]))
