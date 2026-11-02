import copy

import numpy as np
import pandas as pd
import pytest

from creditlab import funnel, pipeline, pricing


def test_ead_factor():
    assert pricing.ead_factor(0.0, 36) == pytest.approx(np.mean([1 - k / 36 for k in range(12)]))
    assert pricing.ead_factor(0.20, 36) > pricing.ead_factor(0.0, 36)      # interest slows amortisation
    assert pricing.ead_factor(0.10, 60) > pricing.ead_factor(0.10, 36)
    v = pricing.ead_factor(np.array([0.1, 0.2]), 60)
    assert v.shape == (2,) and v[1] > v[0]


def test_bands_and_grid(cfg):
    p = cfg["pricing"]
    assert list(pricing.assign_bands([0.0, 0.02, 0.0201, 0.5], p["band_upper_pd"], p["band_names"])) == \
        ["A", "B", "B", "G"]
    apps = pd.DataFrame({"pd": [0.01, 0.01, 0.35], "band": ["A", "A", "G"], "default_12m": [0, 1, 1],
                         "lc_rate": [0.08, 0.09, 0.30], "loan_amnt": [1e4] * 3, "term_months": [36] * 3})
    g = pricing.band_table(apps, cfg)
    cost = p["cost_of_funds"] + p["opex"]
    a = g.loc["A"]
    assert a["el_rate"] == pytest.approx(0.01 * p["lgd"])
    assert a["apr_margin_3%"] == pytest.approx(cost + a["el_rate"] + 0.03)
    assert a["margin_at_observed_default"] == pytest.approx(a["apr_margin_3%"] - cost - 0.5 * p["lgd"])
    assert g.loc["G", "apr_margin_4%"] == p["apr_cap"] and bool(g.loc["G", "cap_binds"])


def test_empirical_lgd():
    df = pd.DataFrame({"loan_status": ["Charged Off", "Charged Off", "Fully Paid"], "funded_amnt": [1000, 1000, 500],
                       "total_rec_prncp": [200, 0, 500], "recoveries": [100, 0, 0],
                       "collection_recovery_fee": [20, 0, 0]})
    out = pricing.empirical_lgd(df)
    assert out["loans"] == 2 and out["lgd"] == pytest.approx(1 - 80 / 1800)


def test_take_up_and_adverse_selection(cfg):
    fp = funnel.FunnelParams.from_config(cfg)
    assert funnel.take_up(0.15, 0.15, fp) == pytest.approx(fp.max_take_up / 2)
    t = funnel.take_up(np.array([0.10, 0.15, 0.20]), 0.15, fp)
    assert t[0] > t[1] > t[2]
    assert funnel.pd_uplift(0.12, 0.15, fp) == 1.0 and funnel.pd_uplift(0.17, 0.15, fp) > 1.0


@pytest.fixture(scope="module")
def apps(parts, cfg):
    from creditlab import models
    m = models.calibrate("scorecard", models.fit_scorecard(parts.train, cfg), parts.valid)
    a = parts.test.copy()
    a["pd"] = m.pd(a)
    a["band"] = pricing.assign_bands(a["pd"], cfg["pricing"]["band_upper_pd"], cfg["pricing"]["band_names"])
    return a


def test_optimiser_respects_constraints(apps, cfg):
    market = apps.groupby("band")["lc_rate"].median().to_dict()
    fp = funnel.FunnelParams.from_config(cfg)
    res = funnel.optimise(apps, market, cfg["pricing"]["band_names"], fp)
    t = res["table"]
    approved = t[t["approved"]]
    assert not approved.empty
    assert (approved["apr"] <= fp.apr_cap + 1e-12).all() and (approved["margin"] >= fp.min_margin - 1e-12).all()
    assert res["portfolio"]["el_rate"] <= fp.max_portfolio_el + 1e-12
    # with a loose appetite the constraint does not bind
    loose = funnel.FunnelParams.from_config({**cfg, "funnel": {**cfg["funnel"], "max_portfolio_el": 1.0}})
    assert funnel.optimise(apps, market, cfg["pricing"]["band_names"], loose)["multiplier"] == 0.0
    tight_cfg = copy.deepcopy(cfg)
    tight_cfg["funnel"]["max_portfolio_el"] = 0.03
    tight = funnel.optimise(apps, market, cfg["pricing"]["band_names"], funnel.FunnelParams.from_config(tight_cfg))
    assert tight["portfolio"]["el_rate"] <= 0.03 + 1e-12
    assert tight["portfolio"]["profit"] <= res["portfolio"]["profit"] + 1e-6


def test_funnel_and_sensitivity(apps, cfg):
    market = apps.groupby("band")["lc_rate"].median().to_dict()
    res = funnel.optimise(apps, market, cfg["pricing"]["band_names"], funnel.FunnelParams.from_config(cfg))
    f = funnel.funnel_summary(apps, res)
    assert f.loc["total", "applicants"] == len(apps)
    assert f.loc["total", "booked"] <= f.loc["total", "approved"] <= f.loc["total", "applicants"]
    s = funnel.sensitivity(apps, market, cfg["pricing"]["band_names"], cfg)
    assert list(s["multiplier"]) == cfg["funnel"]["sensitivity_multipliers"]


def test_lgd_sensitivity_and_appetite_stress(apps, cfg):
    market = apps.groupby("band")["lc_rate"].median().to_dict()
    bands = cfg["pricing"]["band_names"]
    lg = funnel.lgd_sensitivity(apps, market, bands, cfg)
    assert list(lg["lgd"]) == cfg["pricing"]["lgd_sensitivity"]
    assert lg["profit"].is_monotonic_decreasing                       # a higher loss given default never helps
    st = funnel.appetite_stress(apps, market, bands, cfg)
    assert list(st["appetite"]) == cfg["funnel"]["appetite_stress"]
    assert (st["el_rate"] <= st["appetite"] + 1e-9).all()               # every policy meets its own appetite
    assert st["profit"].is_monotonic_decreasing                       # a tighter appetite never adds profit


def test_champion_rule():
    assert pipeline.champion({"low": 0.001})[0] == "gbm"
    assert pipeline.champion({"low": -0.002})[0] == "scorecard"
