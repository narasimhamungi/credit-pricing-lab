"""Synthetic loan file in the Lending Club layout, for tests, CI and offline
runs. Every row is invented. The true default model is known (so tests can
check that the pipeline recovers it), default rates drift upward after 2015
(so out-of-time calibration and PSI have something to find), and the file
mimics the published one's quirks: '%'-suffixed rates, 'Mon-YYYY' dates,
' 36 months' terms and summary rows at the end.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

DATA_END = pd.Timestamp("2019-03-01")
COLUMNS = ["id", "loan_amnt", "funded_amnt", "term", "int_rate", "installment", "grade", "sub_grade", "emp_length",
           "home_ownership", "annual_inc", "verification_status", "issue_d", "loan_status", "purpose", "addr_state",
           "dti", "delinq_2yrs", "earliest_cr_line", "fico_range_low", "fico_range_high", "inq_last_6mths",
           "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies",
           "application_type", "last_pymnt_d", "total_rec_prncp", "recoveries", "collection_recovery_fee"]


def true_logit(df: pd.DataFrame) -> np.ndarray:
    """The data-generating default model (12-month horizon)."""
    year = df["_issue"].dt.year.to_numpy()
    return (-3.1 - 0.028 * (df["_fico"] - 700) + 0.03 * (df["dti"] - 18) + 0.22 * df["inq_last_6mths"]
            + 0.012 * (df["_util"] - 50) + 0.5 * (df["_term"] == 60) + 0.6 * np.log(df["_lti"] / 0.2)
            - 0.25 * (df["home_ownership"] == "MORTGAGE") + 0.12 * np.clip(year - 2015, 0, None)).to_numpy()


def frame(n: int = 60000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range("2013-01-01", "2018-12-01", freq="MS")
    weights = np.linspace(1.0, 3.0, len(months))
    issue = pd.Series(rng.choice(months, size=n, p=weights / weights.sum()))
    fico = np.clip(rng.normal(705, 32, n), 660, 845).round(0) // 5 * 5
    income = np.exp(rng.normal(11.1, 0.5, n)).round(-2)
    amount = (rng.uniform(1000, 40000, n) // 25 * 25)
    term = np.where(rng.random(n) < 0.72, 36, 60)
    df = pd.DataFrame({
        "_issue": issue, "_fico": fico, "_term": term, "_lti": amount / income,
        "_util": np.clip(rng.beta(2.2, 2.4, n) * 100, 0, 120).round(1),
        "dti": np.clip(rng.gamma(4.0, 4.6, n), 0, 60).round(2),
        "inq_last_6mths": rng.poisson(0.7, n),
        "home_ownership": rng.choice(["MORTGAGE", "RENT", "OWN", "ANY"], n, p=[0.49, 0.40, 0.10, 0.01]),
    })
    p = 1 / (1 + np.exp(-true_logit(df)))
    d12 = rng.random(n) < p
    later = (~d12) & (rng.random(n) < p * 1.1)
    m_last = np.where(d12, rng.integers(0, 12, n), np.where(later, rng.integers(12, 30, n), -1))
    lc_pd = 1 / (1 + np.exp(-(true_logit(df) + rng.normal(0, 0.6, n))))
    rate = np.clip(0.055 + 1.6 * lc_pd + rng.normal(0, 0.01, n), 0.053, 0.31)
    grade = pd.cut(rate, [0, 0.08, 0.11, 0.14, 0.18, 0.22, 0.26, 1], labels=list("ABCDEFG")).astype(str)
    out = pd.DataFrame({
        "id": np.arange(1, n + 1) + 10_000_000, "loan_amnt": amount, "funded_amnt": amount,
        "term": [f" {t} months" for t in term], "int_rate": [f"{r * 100:.2f}%" for r in rate],
        "installment": (amount * (rate / 12) / (1 - (1 + rate / 12) ** -term)).round(2),
        "grade": grade, "sub_grade": [g + "3" for g in grade],
        "emp_length": rng.choice(["< 1 year", "1 year", "3 years", "5 years", "8 years", "10+ years", "n/a"], n),
        "home_ownership": df["home_ownership"], "annual_inc": income,
        "verification_status": rng.choice(["Verified", "Source Verified", "Not Verified"], n),
        "issue_d": issue.dt.strftime("%b-%Y"),
        "purpose": rng.choice(["debt_consolidation", "credit_card", "home_improvement", "other", "car",
                               "small_business", "vacation"], n, p=[0.55, 0.22, 0.07, 0.08, 0.03, 0.02, 0.03]),
        "addr_state": rng.choice(["CA", "NY", "TX", "FL", "IL"], n),
        "dti": df["dti"], "delinq_2yrs": rng.poisson(0.3, n),
        "earliest_cr_line": (issue - pd.to_timedelta(rng.integers(36, 360, n) * 30, unit="D")).dt.strftime("%b-%Y"),
        "fico_range_low": fico, "fico_range_high": fico + 4, "inq_last_6mths": df["inq_last_6mths"],
        "open_acc": rng.poisson(11, n) + 1, "pub_rec": rng.poisson(0.2, n),
        "revol_bal": (income * rng.uniform(0.05, 0.5, n)).round(0), "revol_util": [f"{u:.1f}%" for u in df["_util"]],
        "total_acc": rng.poisson(24, n) + 2, "mort_acc": rng.poisson(1.5, n), "pub_rec_bankruptcies": rng.poisson(0.1, n),
        "application_type": np.where(rng.random(n) < 0.97, "Individual", "Joint App"),
    })
    status, last, prin, rec = [], [], [], []
    for i in range(n):
        start = issue.iloc[i]
        if m_last[i] >= 0:
            lp = start + pd.DateOffset(months=int(m_last[i]))
            charged = lp + pd.DateOffset(months=5) <= DATA_END
            status.append("Charged Off" if charged else "Late (31-120 days)")
            last.append(lp if lp <= DATA_END else pd.NaT)
            paid = amount[i] * 0.8 * (m_last[i] + 1) / term[i]
            prin.append(round(paid, 2))
            rec.append(round((amount[i] - paid) * rng.uniform(0.0, 0.25), 2) if charged else 0.0)
        else:
            end = start + pd.DateOffset(months=int(term[i]))
            early = start + pd.DateOffset(months=int(rng.integers(6, term[i])))
            if rng.random() < 0.35 and early <= DATA_END:
                status.append("Fully Paid"), last.append(early), prin.append(amount[i]), rec.append(0.0)
            elif end <= DATA_END:
                status.append("Fully Paid"), last.append(end), prin.append(amount[i]), rec.append(0.0)
            else:
                status.append("Current"), last.append(DATA_END)
                prin.append(round(amount[i] * ((DATA_END - start).days / 30.4) / term[i] * 0.85, 2)), rec.append(0.0)
    out["loan_status"] = status
    out["last_pymnt_d"] = [d.strftime("%b-%Y") if pd.notna(d) else "" for d in last]
    out["total_rec_prncp"] = prin
    out["recoveries"] = rec
    out["collection_recovery_fee"] = (out["recoveries"] * 0.18).round(2)
    return out[COLUMNS]


def write(path: Path, n: int = 60000, seed: int = 11) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = frame(n, seed)
    opener = gzip.open if path.suffix == ".gz" else open
    # "\n" throughout: pandas otherwise ends rows with os.linesep ("\r\n" on Windows), and a file that mixes
    # "\r\n" rows with "\n" summary lines stops DuckDB's CSV reader. The CSV text is the same on every platform.
    with opener(path, "wt", encoding="utf-8", newline="") as fh:
        df.to_csv(fh, index=False, lineterminator="\n")
        fh.write("\nTotal amount funded in policy code 1: 0\nTotal amount funded in policy code 2: 0\n")
    return path
