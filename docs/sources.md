# Sources

| Source | What is used | Licence | Checked |
|---|---|---|---|
| Lending Club accepted loans 2007-2018Q4, Kaggle mirror ([wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)), file `accepted_2007_to_2018Q4.csv.gz` | Loan-level application fields, loan status and payments, staged by `src/creditlab/sql/01_staging.sql` | CC0 1.0 (public domain dedication), as stated on the dataset page | 2026-11-02 |

The dedication is the uploader's. It covers the rights the uploader holds and, as the CC0 deed says, does not affect any rights others may have in the work or in how it is used. The repository therefore still commits no loan-level data: `data/` is git-ignored, and `outputs/` holds aggregate tables and charts only.
