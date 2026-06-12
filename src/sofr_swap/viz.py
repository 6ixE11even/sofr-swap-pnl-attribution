"""Plots: the two-day SOFR curve and the P&L attribution waterfall."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sofr_swap.curve import SofrCurve


def plot_curves(curve0: SofrCurve, curve1: SofrCurve, out_path: str | Path,
                labels=("day 0", "day 1")) -> None:
    t = np.linspace(0.5, curve0.pillar_times().max(), 120)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(t, curve0.zero_rate(t) * 100, label=labels[0], lw=1.8)
    ax.plot(t, curve1.zero_rate(t) * 100, label=labels[1], lw=1.8, ls="--")
    ax.set_title("SOFR zero curve", fontweight="bold")
    ax.set_xlabel("maturity (years)")
    ax.set_ylabel("zero rate (%)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    _save(fig, out_path)


def plot_waterfall(buckets: dict[str, float], out_path: str | Path) -> None:
    """Bridge from 0 to total P&L through the attribution buckets."""
    order = ["carry", "roll_down", "level", "slope", "curvature", "residual"]
    vals = [buckets[k] for k in order]
    fig, ax = plt.subplots(figsize=(9.5, 5))

    running = 0.0
    for i, (name, v) in enumerate(zip(order, vals)):
        color = "#2c7a4b" if v >= 0 else "#c0392b"
        ax.bar(i, v, bottom=running, color=color, edgecolor="black", linewidth=0.4)
        running += v
    ax.bar(len(order), running, color="#34495e", edgecolor="black", linewidth=0.4)

    ax.set_xticks(range(len(order) + 1))
    ax.set_xticklabels([*[o.replace("_", "-") for o in order], "TOTAL"], rotation=20)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Daily P&L attribution", fontweight="bold")
    ax.set_ylabel("P&L ($)")
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, out_path)


def _save(fig, out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
