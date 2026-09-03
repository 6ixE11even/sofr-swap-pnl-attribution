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


def plot_exposure_profile(times, prof, path):
    """EE, EPE and the 97.5% PFE over the life of the book."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.fill_between(times, 0, prof["pfe"] / 1e6, color="#c6dbef", label="PFE 97.5%")
    ax.plot(times, prof["epe"] / 1e6, color="#1f77b4", lw=1.8, label="EPE (expected positive)")
    ax.plot(times, prof["ee"] / 1e6, color="#333333", lw=1.4, ls="--", label="EE (expected)")
    ax.plot(times, prof["ene"] / 1e6, color="#d62728", lw=1.2, ls=":", label="ENE (expected negative)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("years from today")
    ax.set_ylabel("$mm")
    ax.set_title("Counterparty exposure profile")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cva_decomposition(runs, path):
    """CVA with the rate/credit link as observed, against the same book with it broken."""
    labels = list(runs)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    x = np.arange(len(labels))
    for i, key in enumerate(("cva", "dva", "bcva")):
        ax.bar(x + (i - 1) * 0.26, [runs[l][key] / 1e3 for l in labels], 0.25,
               label=key.upper(), color=["#d62728", "#2ca02c", "#1f77b4"][i])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("$k")
    ax.set_title("Rates and credit moving together is worth real money")
    ax.legend(fontsize=8)

    # The two runs share their curve paths by construction, so their exposure profiles
    # are the same line to the dollar. What differs is which states of the world carry
    # default probability, so plot where the CVA is actually earned instead.
    ax = axes[1]
    for label, colour, style in zip(labels, ("#1f77b4", "#7f7f7f"), ("-", "--")):
        r = runs[label]
        ax.plot(r["sim"].times, np.cumsum(r["by_time"]) / 1e3, style, color=colour,
                label=f"{label}  (${r['cva'] / 1e3:,.0f}k)")
    ax.fill_between(runs[labels[0]]["sim"].times,
                    np.cumsum(runs[labels[1]]["by_time"]) / 1e3,
                    np.cumsum(runs[labels[0]]["by_time"]) / 1e3,
                    color="#d62728", alpha=0.15, lw=0)
    ax.set_xlabel("years from today")
    ax.set_ylabel("cumulative CVA ($k)")
    ax.set_title("Same exposure paths; the shaded gap is the coupling")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
