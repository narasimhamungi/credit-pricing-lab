"""Charts (PNG): reliability on the out-of-time window, pricing against the
market, profit against APR by band, and the price-sensitivity sensitivity.

Three validated categorical slots (blue, orange, aqua); text in neutral inks;
hairline solid grid; one y-axis per panel (two measures get two panels)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e6e3", "#ffffff"
FONT = "Arial" if any(f.name == "Arial" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({
    "font.family": FONT, "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160, "savefig.facecolor": SURFACE,
    "legend.frameon": False,
})


def _done(fig, path: Path, synthetic: bool) -> Path:
    if synthetic:
        fig.text(0.99, 0.01, "SYNTHETIC DATA", ha="right", va="bottom", fontsize=8, color="#c00000", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def reliability(curves: dict[str, pd.DataFrame], path: Path, synthetic: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    top = max(max(c["mean_pd"].max(), c["default_rate"].max()) for c in curves.values()) * 108
    ax.plot([0, top], [0, top], color=MUTED, linewidth=1, label="Perfect calibration")
    for (name, c), color in zip(curves.items(), SERIES):
        ax.plot(c["mean_pd"] * 100, c["default_rate"] * 100, color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=name)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("Mean predicted 12-month PD by decile (%)")
    ax.set_ylabel("Observed 12-month default rate (%)")
    ax.legend(loc="upper left", fontsize=8)
    return _done(fig, path, synthetic)


def pricing_vs_market(grid: pd.DataFrame, margin_col: str, path: Path, synthetic: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    x = np.arange(len(grid))
    ax.plot(x, grid[margin_col] * 100, color=SERIES[0], linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, label="Model APR (target margin)")
    ax.plot(x, grid["lc_median_rate"] * 100, color=SERIES[1], linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, label="Lending Club median rate")
    if "breakeven_apr_observed" in grid:
        ax.plot(x, grid["breakeven_apr_observed"] * 100, color=SERIES[2], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, label="Break-even APR at the observed default rate")
    ax.set_xticks(x, grid.index)
    ax.set_xlabel("Risk band (calibrated PD)")
    ax.set_ylabel("Annual rate (%)")
    ax.legend(loc="upper left", fontsize=8)
    return _done(fig, path, synthetic)


def profit_curves(curves: dict[str, pd.DataFrame], chosen: pd.DataFrame, bands: list[str], path: Path,
                  synthetic: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for band, color in zip(bands, SERIES):
        if band not in curves:
            continue
        c = curves[band]
        ax.plot(c["apr"] * 100, c["profit"] / 1e3, color=color, linewidth=2, label=f"Band {band}")
        if band in chosen.index and chosen.loc[band, "apr"] is not None and not pd.isna(chosen.loc[band, "apr"]):
            r = chosen.loc[band]
            ax.plot([r["apr"] * 100], [r["profit"] / 1e3], marker="o", markersize=7, color=color,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.annotate(f"{r['apr'] * 100:.2f}%", (r["apr"] * 100, r["profit"] / 1e3), xytext=(5, 6),
                        textcoords="offset points", fontsize=8, color=INK2)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("APR (%)")
    ax.set_ylabel("Expected annual profit ($ thousands)")
    ax.legend(loc="lower left", fontsize=8)
    return _done(fig, path, synthetic)


def sensitivity(table: pd.DataFrame, path: Path, synthetic: bool = False) -> Path:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.2))
    x = table["multiplier"]
    a1.plot(x, table["profit"] / 1e6, color=SERIES[0], linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE)
    a1.set_xlabel("Price sensitivity (x base assumption)")
    a1.set_ylabel("Portfolio profit ($m a year)")
    a2.plot(x, table["avg_apr"] * 100, color=SERIES[0], linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE)
    a2.set_xlabel("Price sensitivity (x base assumption)")
    a2.set_ylabel("Balance-weighted APR (%)")
    for ax in (a1, a2):
        ax.axvline(1.0, color=MUTED, linewidth=1)
    return _done(fig, path, synthetic)
