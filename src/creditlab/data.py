"""Load the raw file into DuckDB and build the modelling table with the SQL in
creditlab/sql: typed staging, the 12-month default label, application-time
features. Also the leakage guard and the out-of-time split."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from creditlab.settings import SQL_DIR

# Known only after origination, or Lending Club's own risk view and price.
FORBIDDEN = {
    "grade", "sub_grade", "lc_grade", "int_rate", "lc_rate", "installment", "funded_amnt_inv", "total_pymnt",
    "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
    "last_fico_range_high", "last_fico_range_low", "out_prncp", "out_prncp_inv", "loan_status",
    "months_to_last_payment", "default_12m", "hardship_flag", "debt_settlement_flag", "settlement_status",
}


class DataError(ValueError):
    pass


def check_features(cfg: dict) -> list[str]:
    feats = list(cfg["features"]["numeric"]) + list(cfg["features"]["categorical"])
    leaked = sorted(set(feats) & FORBIDDEN)
    if leaked:
        raise DataError(f"features known only after origination (or Lending Club's own pricing): {leaked}")
    return feats


def _sql(name: str, **params) -> str:
    return (SQL_DIR / name).read_text(encoding="utf-8").format(**params)


def build_table(raw_path: str | Path, cfg: dict, con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Run the SQL pipeline on the raw CSV (plain or gzipped) and return the
    labelled feature table for loans inside the split windows whose 12-month
    window had closed by the snapshot date."""
    check_features(cfg)
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise DataError(f"raw file not found: {raw_path}. Download the Lending Club accepted-loans file (see README) "
                        "or run `creditlab synthetic` for test data")
    con = con or duckdb.connect()
    path = str(raw_path).replace("'", "''")
    con.execute(f"CREATE OR REPLACE VIEW raw AS SELECT * FROM read_csv('{path}', header = true, all_varchar = true, "
                "ignore_errors = true, null_padding = true)")
    con.execute(_sql("01_staging.sql"))
    data_end = cfg["data"].get("data_end")
    if not data_end:
        found = con.execute("SELECT max(last_pymnt_d) FROM loans").fetchone()[0]
        if found is None:
            raise DataError("no last-payment dates in the file; set data.data_end")
        data_end = found
    data_end = pd.Timestamp(data_end).date().isoformat()
    t = cfg["target"]
    statuses = ", ".join("'" + s.replace("'", "''") + "'" for s in t["default_statuses"])
    con.execute(_sql("02_label.sql", default_statuses=statuses, horizon=int(t["horizon_months"]),
                     last_paid_month=int(t["horizon_months"]) - 1, data_end=data_end))
    con.execute(_sql("03_features.sql"))
    sp = cfg["split"]
    start, end = sp["train"][0], sp["test"][1]
    df = con.execute(f"SELECT * FROM features WHERE window_complete AND issue_d BETWEEN DATE '{start}' AND "
                     f"DATE '{end}'").df()
    df.attrs["data_end"] = data_end
    df.attrs["rows_staged"] = con.execute("SELECT count(*) FROM loans").fetchone()[0]
    df.attrs["rows_incomplete_window"] = con.execute(
        f"SELECT count(*) FROM labelled WHERE NOT window_complete AND issue_d BETWEEN DATE '{start}' AND "
        f"DATE '{end}'").fetchone()[0]
    return df


@dataclass
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        rows = []
        for name in ("train", "valid", "test"):
            d = getattr(self, name)
            rows.append({"split": name, "from": d["issue_d"].min(), "to": d["issue_d"].max(), "loans": len(d),
                         "defaults": int(d["default_12m"].sum()), "default_rate": d["default_12m"].mean()})
        return pd.DataFrame(rows).set_index("split")


def split(df: pd.DataFrame, cfg: dict) -> Split:
    issue = pd.to_datetime(df["issue_d"])
    parts = {}
    for name in ("train", "valid", "test"):
        lo, hi = (pd.Timestamp(x) for x in cfg["split"][name])
        parts[name] = df[(issue >= lo) & (issue <= hi)].reset_index(drop=True)
    windows = [tuple(pd.Timestamp(x) for x in cfg["split"][n]) for n in ("train", "valid", "test")]
    if not (windows[0][1] < windows[1][0] and windows[1][1] < windows[2][0]):
        raise DataError("split windows must be in time order and must not overlap")
    for name, part in parts.items():
        if part.empty:
            raise DataError(f"no loans in the {name} window {cfg['split'][name]}")
        if part["default_12m"].nunique() < 2:
            raise DataError(f"the {name} window has only one outcome class")
    return Split(parts["train"], parts["valid"], parts["test"])
