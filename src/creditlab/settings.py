"""Paths and the YAML settings."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(os.environ.get("CREDITLAB_CONFIG", PROJECT_ROOT / "config" / "settings.yml"))
OUTPUT_DIR = Path(os.environ.get("CREDITLAB_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
SQL_DIR = Path(__file__).resolve().parent / "sql"


def load(path: Path = CONFIG_PATH) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    raw = os.environ.get("CREDITLAB_RAW_PATH")
    if raw:
        cfg["data"]["raw_path"] = raw
    p = Path(cfg["data"]["raw_path"])
    if not p.is_absolute():
        cfg["data"]["raw_path"] = str(PROJECT_ROOT / p)
    return cfg
