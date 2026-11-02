"""End to end: SQL features -> out-of-time split -> scorecard and GBM ->
calibration and validation -> champion -> EL and APR grid -> funnel
optimisation and sensitivity -> CSVs, charts, metrics.json, report.md."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from creditlab import charts, data, funnel, metrics, models, pricing, report


def champion(gbm_vs_scorecard: dict) -> tuple[str, str]:
    """Keep the scorecard unless the GBM's out-of-time AUC is better with the
    whole bootstrap interval above zero."""
    if gbm_vs_scorecard["low"] > 0:
        return "gbm", "GBM's out-of-time AUC gain is significant (95% interval above zero)"
    return "scorecard", "GBM's out-of-time AUC gain is not significant, so the simpler, explainable scorecard is kept"


def run(raw_path: str | Path, out_dir: str | Path, cfg: dict, synthetic: bool = False,
        export_scores: str | Path | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = data.build_table(raw_path, cfg)
    sp = data.split(table, cfg)
    sc = models.fit_scorecard(sp.train, cfg)
    gb = models.fit_gbm(sp.train, cfg)
    fitted = {"scorecard": models.calibrate("scorecard", sc, sp.valid), "gbm": models.calibrate("gbm", gb, sp.valid)}
    y = sp.test["default_12m"].to_numpy()
    evaluation, rel = {}, {}
    for name, m in fitted.items():
        raw_test, pd_test = m.raw(sp.test), m.pd(sp.test)
        raw_train = m.raw(sp.train)
        psi_value = metrics.psi(raw_train, raw_test)
        evaluation[name] = {
            "test": {**metrics.discrimination(y, raw_test), **metrics.calibration(y, pd_test)},
            "valid": metrics.discrimination(sp.valid["default_12m"], m.raw(sp.valid)),
            "train": metrics.discrimination(sp.train["default_12m"], raw_train),
            "uncalibrated_test": metrics.calibration(y, raw_test),
            "score_psi_train_to_test": psi_value, "psi_label": metrics.psi_label(psi_value),
        }
        rel[name] = metrics.reliability(y, pd_test)
    diff = metrics.bootstrap_auc_difference(y, fitted["scorecard"].raw(sp.test), fitted["gbm"].raw(sp.test),
                                            cfg["models"]["bootstrap_draws"], cfg["models"]["random_state"])
    champ, reason = champion(diff)
    csi = {f: metrics.psi(pd.to_numeric(sp.train[f], errors="coerce"), pd.to_numeric(sp.test[f], errors="coerce"))
           for f in cfg["features"]["numeric"]}
    lgd_emp = pricing.empirical_lgd(sp.train)
    if export_scores:
        _export_scores(sp, fitted, Path(export_scores))
    cfg = copy.deepcopy(cfg)
    p = cfg["pricing"]
    assumed_lgd = p["lgd"]
    if p.get("lgd_source", "assumption") == "empirical" and lgd_emp.get("lgd") is not None:
        p["lgd"], lgd_basis = lgd_emp["lgd"], "empirical (training window)"
    else:
        lgd_basis = "assumption"

    apps = sp.test.copy()
    apps["pd"] = fitted[champ].pd(apps)
    apps["band"] = pricing.assign_bands(apps["pd"], p["band_upper_pd"], p["band_names"])
    grid = pricing.band_table(apps, cfg)
    market = apps.groupby("band")["lc_rate"].median().to_dict()
    for b in p["band_names"]:      # bands with no Lending Club rate fall back to the model's base-margin APR
        if b in grid.index and (b not in market or pd.isna(market[b])):
            market[b] = grid.loc[b, f"apr_margin_{p['target_margins'][len(p['target_margins']) // 2]:.0%}"]
    fp = funnel.FunnelParams.from_config(cfg)
    opt = funnel.optimise(apps, market, p["band_names"], fp)
    fun = funnel.funnel_summary(apps, opt)
    sens = funnel.sensitivity(apps, market, p["band_names"], cfg)
    lgd_sens = funnel.lgd_sensitivity(apps, market, p["band_names"], cfg)
    stress = funnel.appetite_stress(apps, market, p["band_names"], cfg)

    files = {}
    files["iv"] = out / "information_value.csv"
    sc.binner.iv_table().rename("iv").to_frame().to_csv(files["iv"])
    files["points"] = out / "scorecard_points.csv"
    sc.points_table().to_csv(files["points"], index=False)
    files["reliability"] = out / "reliability_test.csv"
    pd.concat({k: v for k, v in rel.items()}, names=["model", "decile"]).to_csv(files["reliability"])
    files["grid"] = out / "pricing_grid.csv"
    grid.to_csv(files["grid"])
    cols = ["apr", "market_rate", "mean_pd", "take_up", "booked_loans", "booked_amount", "booked_balance", "revenue",
            "funding_and_opex", "expected_loss", "profit", "margin", "el_rate", "approved"]
    files["optimal"] = out / "optimal_pricing.csv"
    opt["table"][[c for c in cols if c in opt["table"]]].to_csv(files["optimal"])
    files["funnel"] = out / "funnel.csv"
    fun.to_csv(files["funnel"])
    files["sensitivity"] = out / "price_sensitivity.csv"
    sens.to_csv(files["sensitivity"], index=False)
    files["lgd_sensitivity"] = out / "lgd_sensitivity.csv"
    lgd_sens.to_csv(files["lgd_sensitivity"], index=False)
    files["appetite_stress"] = out / "appetite_stress.csv"
    stress.to_csv(files["appetite_stress"], index=False)
    mid_margin = f"apr_margin_{p['target_margins'][len(p['target_margins']) // 2]:.0%}"
    files["chart_reliability"] = charts.reliability({"Scorecard": rel["scorecard"], "Gradient boosting": rel["gbm"]},
                                                    out / "reliability_out_of_time.png", synthetic)
    files["chart_pricing"] = charts.pricing_vs_market(grid, mid_margin, out / "pricing_vs_market.png", synthetic)
    show = [b for b in p["band_names"][1::2]][:3]
    files["chart_profit"] = charts.profit_curves(opt["curves"], opt["table"], show, out / "profit_vs_apr.png",
                                                 synthetic)
    files["chart_sensitivity"] = charts.sensitivity(sens, out / "price_sensitivity.png", synthetic)

    results = {
        "synthetic": synthetic, "data": {"raw_path": Path(raw_path).name, "data_end": table.attrs.get("data_end"),
                                         "rows_staged": table.attrs.get("rows_staged"),
                                         "split": _records(sp.summary())},
        "features": {"numeric": cfg["features"]["numeric"], "categorical": cfg["features"]["categorical"],
                     "scorecard_selected": sc.selected, "information_value": sc.binner.iv_table().to_dict(),
                     "feature_psi_train_to_test": csi},
        "models": evaluation, "auc_difference_gbm_minus_scorecard": diff,
        "champion": {"model": champ, "reason": reason},
        "lgd": {"used": p["lgd"], "basis": lgd_basis, "assumption": assumed_lgd, "empirical_train": lgd_emp},
        "pricing": {"grid": _records(grid), "cost_of_funds": p["cost_of_funds"], "opex": p["opex"],
                    "target_margins": p["target_margins"], "apr_cap": p["apr_cap"]},
        "funnel": {"assumptions": cfg["funnel"], "approved_bands": opt["approved"],
                   "lagrange_multiplier": opt["multiplier"], "portfolio": opt["portfolio"],
                   "bands": _records(opt["table"][[c for c in cols if c in opt["table"]]]),
                   "summary": _records(fun)},
        "price_sensitivity": sens.to_dict(orient="records"),
        "lgd_sensitivity": lgd_sens.to_dict(orient="records"),
        "appetite_stress": stress.to_dict(orient="records"),
        "files": {k: Path(v).name for k, v in files.items()},
    }
    (out / "metrics.json").write_text(json.dumps(results, indent=1, default=_json), encoding="utf-8")
    (out / "report.md").write_text(report.markdown(results), encoding="utf-8")
    results["files"]["metrics"] = "metrics.json"
    results["files"]["report"] = "report.md"
    return results


def _export_scores(sp, fitted: dict, path: Path) -> None:
    """Loan-level scores for independent validation (mv-lab): one row per loan in each window with the outcome,
    both models' raw scores and calibrated PDs. Loan-level, so it belongs under data/ (git-ignored)."""
    frames = []
    for name, part in (("train", sp.train), ("valid", sp.valid), ("test", sp.test)):
        f = pd.DataFrame({"split": name, "issue_d": pd.to_datetime(part["issue_d"]).dt.date,
                          "default_12m": part["default_12m"].astype(int).to_numpy()})
        for m, model in fitted.items():
            f[f"{m}_raw"], f[f"{m}_pd"] = model.raw(part), model.pd(part)
        frames.append(f)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)


def _records(df: pd.DataFrame) -> dict:
    return {str(k): {c: _json(v) for c, v in row.items()} for k, row in df.to_dict(orient="index").items()}


def _json(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.date().isoformat()
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if isinstance(v, float) and np.isnan(v):
        return None
    return v
