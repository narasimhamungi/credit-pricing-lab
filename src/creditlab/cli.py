"""Command line: creditlab <command>.

  synthetic   write a synthetic loan file in the Lending Club layout
  features    build the SQL feature table and print the split summary
  run         the full pipeline: models, validation, pricing, funnel, report
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from creditlab import settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="creditlab", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("synthetic")
    p.add_argument("--out", default="data/synthetic/loans.csv.gz")
    p.add_argument("--rows", type=int, default=60000)
    p = sub.add_parser("features")
    p.add_argument("--raw", help="raw Lending Club CSV (default: data.raw_path in config/settings.yml)")
    p.add_argument("--parquet", help="also write the feature table to this Parquet file")
    p = sub.add_parser("run")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--raw", help="raw Lending Club CSV (default: data.raw_path in config/settings.yml)")
    src.add_argument("--synthetic", action="store_true", help="generate and use the synthetic file")
    p.add_argument("--out", help="output folder (default outputs/, or outputs/synthetic/)")
    p.add_argument("--export-scores", help="also write loan-level scores and PDs per window to this Parquet file "
                                           "(for independent validation; keep it under data/, which is git-ignored)")
    args = parser.parse_args(argv)
    try:
        cfg = settings.load()
        if args.cmd == "synthetic":
            from creditlab import synthetic
            path = synthetic.write(Path(args.out), args.rows)
            print(f"synthetic loan file: {path}")
            return 0
        if args.cmd == "features":
            from creditlab import data
            table = data.build_table(args.raw or cfg["data"]["raw_path"], cfg)
            if args.parquet:
                table.to_parquet(args.parquet, index=False)
            print(data.split(table, cfg).summary().to_string())
            return 0
        from creditlab import pipeline
        if args.synthetic:
            from creditlab import synthetic
            out = Path(args.out) if args.out else settings.OUTPUT_DIR / "synthetic"
            raw = synthetic.write(out / "synthetic_loans.csv.gz")
        else:
            out = Path(args.out) if args.out else settings.OUTPUT_DIR
            raw = args.raw or cfg["data"]["raw_path"]
        res = pipeline.run(raw, out, cfg, synthetic=args.synthetic, export_scores=args.export_scores)
        champ = res["champion"]["model"]
        t = res["models"][champ]["test"]
        print(f"champion {champ}: out-of-time AUC {t['auc']:.3f}, Gini {t['gini']:.3f}, KS {t['ks']:.3f}")
        print(f"approved bands {''.join(res['funnel']['approved_bands'])}; report: {out / res['files']['report']}")
        return 0
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("CREDITLAB_DEBUG"):
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
