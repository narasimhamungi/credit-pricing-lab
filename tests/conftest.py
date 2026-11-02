"""Fixtures: a small synthetic loan file (invented data) and the settings."""
from __future__ import annotations

import pytest

from creditlab import data, settings, synthetic


@pytest.fixture(scope="session")
def cfg() -> dict:
    return settings.load()


@pytest.fixture(scope="session")
def raw_file(tmp_path_factory):
    return synthetic.write(tmp_path_factory.mktemp("raw") / "loans.csv.gz", n=20000, seed=3)


@pytest.fixture(scope="session")
def table(raw_file, cfg):
    return data.build_table(raw_file, cfg)


@pytest.fixture(scope="session")
def parts(table, cfg):
    return data.split(table, cfg)
