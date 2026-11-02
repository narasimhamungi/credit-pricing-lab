import copy
import gzip
import os

import pandas as pd
import pytest

from creditlab import data, synthetic

HEADER = synthetic.COLUMNS


def raw_rows(tmp_path, rows: list[dict]):
    """A tiny raw file in the published layout; unspecified columns blank."""
    base = {c: "" for c in HEADER}
    base.update({"loan_amnt": "10000", "funded_amnt": "10000", "term": " 36 months", "int_rate": "12.50%",
                 "annual_inc": "60000", "dti": "15", "fico_range_low": "700", "fico_range_high": "704",
                 "earliest_cr_line": "Jan-2005", "home_ownership": "RENT", "revol_util": "40.5%",
                 "emp_length": "5 years", "total_rec_prncp": "0", "recoveries": "0", "collection_recovery_fee": "0"})
    df = pd.DataFrame([{**base, **r} for r in rows])[HEADER]
    path = tmp_path / "rows.csv"
    df.to_csv(path, index=False, lineterminator="\n")
    with open(path, "a", encoding="utf-8", newline="") as fh:
        fh.write("\nTotal amount funded in policy code 1: 123\n")
    return path


def test_label_definition(tmp_path, cfg):
    rows = [
        {"id": "1", "issue_d": "Jan-2014", "loan_status": "Charged Off", "last_pymnt_d": "Nov-2014"},  # month 10
        {"id": "2", "issue_d": "Jan-2014", "loan_status": "Charged Off", "last_pymnt_d": "Dec-2014"},  # month 11
        {"id": "3", "issue_d": "Jan-2014", "loan_status": "Charged Off", "last_pymnt_d": "Jan-2015"},  # month 12
        {"id": "4", "issue_d": "Jan-2014", "loan_status": "Charged Off", "last_pymnt_d": ""},          # never paid
        {"id": "5", "issue_d": "Jan-2014", "loan_status": "Fully Paid", "last_pymnt_d": "Mar-2014"},
        {"id": "6", "issue_d": "Jan-2014", "loan_status": "Does not meet the credit policy. Status:Charged Off",
         "last_pymnt_d": "May-2014"},
        {"id": "7", "issue_d": "Jun-2018", "loan_status": "Current", "last_pymnt_d": "Mar-2019"},     # window open
        {"id": "8", "issue_d": "Jan-2014", "loan_status": "Late (31-120 days)", "last_pymnt_d": "Feb-2014"},
    ]
    c = copy.deepcopy(cfg)
    c["split"] = {"train": ["2013-01-01", "2016-12-31"], "valid": ["2017-01-01", "2017-12-31"],
                  "test": ["2018-01-01", "2018-12-31"]}
    df = data.build_table(raw_rows(tmp_path, rows), c).set_index("id")
    assert df.attrs["rows_staged"] == 8                      # the summary footer is dropped
    assert df.attrs["rows_incomplete_window"] == 1 and "7" not in df.index
    assert df.loc["1", "default_12m"] == 1 and df.loc["2", "default_12m"] == 1
    assert df.loc["3", "default_12m"] == 0                   # first missed payment in month 13
    assert df.loc["4", "default_12m"] == 1 and df.loc["6", "default_12m"] == 1 and df.loc["8", "default_12m"] == 1
    assert df.loc["5", "default_12m"] == 0
    r = df.loc["1"]
    assert r["lc_rate"] == pytest.approx(0.125) and r["revol_util"] == pytest.approx(40.5)
    assert r["fico"] == 702 and r["term_months"] == 36 and r["emp_years"] == 5
    assert r["credit_history_years"] == pytest.approx(9.0)


def test_feature_cleaning(tmp_path, cfg):
    rows = [{"id": str(i), "issue_d": "Jan-2014", "loan_status": "Fully Paid", "last_pymnt_d": "Jan-2016", **extra}
            for i, extra in enumerate([{"emp_length": "10+ years", "home_ownership": "ANY", "dti": "999"},
                                       {"emp_length": "< 1 year", "home_ownership": "mortgage", "purpose": "wedding"},
                                       {"emp_length": "n/a", "home_ownership": "NONE"}])]
    df = data.build_table(raw_rows(tmp_path, rows), cfg).set_index("id")
    assert df["emp_years"].tolist()[:2] == [10.0, 0.0] and pd.isna(df.loc["2", "emp_years"])
    assert df.loc["0", "home_ownership"] == "OTHER" and df.loc["1", "home_ownership"] == "MORTGAGE"
    assert pd.isna(df.loc["0", "dti"])                       # implausible DTI dropped
    assert df.loc["1", "purpose_group"] == "other"


def test_leakage_guard(cfg):
    c = copy.deepcopy(cfg)
    c["features"]["numeric"] = c["features"]["numeric"] + ["int_rate", "recoveries"]
    with pytest.raises(data.DataError, match="int_rate"):
        data.check_features(c)


def test_split_is_out_of_time(parts, cfg):
    assert parts.train["issue_d"].max() < parts.valid["issue_d"].min()
    assert parts.valid["issue_d"].max() < parts.test["issue_d"].min()
    s = parts.summary()
    assert (s["loans"] > 0).all() and (s["default_rate"] > 0).all()
    c = copy.deepcopy(cfg)
    c["split"]["valid"] = ["2015-06-01", "2016-12-31"]
    with pytest.raises(data.DataError, match="time order"):
        data.split(pd.concat([parts.train, parts.valid, parts.test]), c)


def test_missing_file_message(tmp_path, cfg):
    with pytest.raises(data.DataError, match="creditlab synthetic"):
        data.build_table(tmp_path / "nope.csv.gz", cfg)


def test_synthetic_file_has_unix_line_endings_on_every_platform(tmp_path, cfg, monkeypatch):
    """pandas ends rows with os.linesep, "\\r\\n" on Windows; the writer must not inherit it."""
    monkeypatch.setattr(os, "linesep", "\r\n")
    path = synthetic.write(tmp_path / "win.csv.gz", n=3000, seed=5)
    with gzip.open(path, "rb") as fh:
        assert b"\r" not in fh.read()
    assert len(data.build_table(path, cfg)) > 0
