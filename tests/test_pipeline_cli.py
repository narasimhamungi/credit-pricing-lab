import json

from creditlab import cli, pipeline


def test_pipeline_outputs(raw_file, cfg, tmp_path):
    res = pipeline.run(raw_file, tmp_path, cfg, synthetic=True, export_scores=tmp_path / "scores.parquet")
    import pandas as pd
    sc = pd.read_parquet(tmp_path / "scores.parquet")
    assert set(sc["split"]) == {"train", "valid", "test"} and sc[["gbm_pd", "scorecard_pd"]].notna().all().all()
    assert set(sc.columns) >= {"issue_d", "default_12m", "gbm_raw", "scorecard_raw"}
    for key in ("reliability", "grid", "optimal", "funnel", "sensitivity", "chart_reliability", "chart_pricing",
                "chart_profit", "chart_sensitivity", "metrics", "report", "points", "iv", "lgd_sensitivity",
                "appetite_stress"):
        assert (tmp_path / res["files"][key]).exists(), key
    m = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert set(m["models"]) == {"scorecard", "gbm"}
    assert m["models"]["scorecard"]["test"]["auc"] > 0.6
    assert m["champion"]["model"] in ("scorecard", "gbm")
    assert m["lgd"]["basis"].startswith("empirical") and m["lgd"]["used"] == m["lgd"]["empirical_train"]["lgd"]
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert report.startswith("# Credit pricing lab") and "SYNTHETIC DATA" in report
    assert "## Loss-appetite stress" in report and "## LGD sensitivity" in report


def test_cli(tmp_path, capsys):
    out = tmp_path / "s.csv.gz"
    assert cli.main(["synthetic", "--out", str(out), "--rows", "3000"]) == 0
    assert cli.main(["features", "--raw", str(out)]) == 0
    assert "train" in capsys.readouterr().out
    assert cli.main(["features", "--raw", str(tmp_path / "missing.csv")]) == 2
    assert "creditlab synthetic" in capsys.readouterr().err
