"""
Attribute one day of P&L on the swap book.

    uv run python scripts/run_attribution.py                 # the two most recent days
    uv run python scripts/run_attribution.py --worst-day     # the biggest move in the sample
    uv run python scripts/run_attribution.py --date 2020-03-09

Curves are bootstrapped from real Treasury constant-maturity par yields pulled from
FRED, on two consecutive business days. The book is marked on each, and the change is
split into carry, roll-down and the level/slope/curvature of the curve move.

The book itself is hypothetical - ten swaps, a net receiver position - because nobody
publishes a real dealer's positions. The market data is not.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sofr_swap.attribution import attribute            # noqa: E402
from sofr_swap.curve import SofrCurve                  # noqa: E402
from sofr_swap.instruments import Swap                 # noqa: E402
from sofr_swap.market import consecutive_pairs, rates_panel  # noqa: E402
from sofr_swap.pricing import dv01, swap_npv           # noqa: E402
from sofr_swap.viz import plot_curves, plot_waterfall  # noqa: E402

ONE_DAY = 1.0 / 365.0
RESULTS = ROOT / "results"


def load_curves(date: str | None, worst_day: bool) -> tuple[SofrCurve, SofrCurve, list[str]]:
    """Two consecutive real curves: the pair ending on `date`, the latest, or the
    single largest one-day curve move in the sample."""
    rates = rates_panel()
    tenors = np.array(rates.columns, dtype=float)
    adjacent = consecutive_pairs(rates).to_numpy()

    if worst_day:
        # "Largest" in the sense that matters to a rates book: the biggest move in the
        # ten-year, which is the pillar the book's DV01 is concentrated in. Only rows
        # whose predecessor is the previous business day are eligible - see
        # consecutive_pairs.
        move = np.abs(rates[10].diff().to_numpy())
        move[~adjacent] = -np.inf
        i = int(np.nanargmax(move))
    elif date is not None:
        i = int(rates.index.get_indexer([pd.Timestamp(date)], method="nearest")[0])
    else:
        i = len(rates) - 1

    if not adjacent[i]:
        raise ValueError(f"{rates.index[i]:%Y-%m-%d} has no preceding business day in the "
                         "pillar history; the curve is incomplete around it")
    pair = rates.iloc[[i - 1, i]]
    curves = [SofrCurve.bootstrap(tenors, row.to_numpy()) for _, row in pair.iterrows()]
    move = (pair.iloc[1] - pair.iloc[0]) * 1e4
    print("curve move (bp): " + "  ".join(f"{int(t)}y {m:+.0f}" for t, m in move.items()))
    return curves[0], curves[1], [f"{d:%Y-%m-%d}" for d in pair.index]


def load_book() -> list[Swap]:
    p = pd.read_csv(ROOT / "data" / "positions.csv")
    return [Swap(r.notional, r.fixed_rate, r.effective, r.maturity, r.side, r.trade_id)
            for r in p.itertuples()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", help="value the book on this date and the day before it")
    ap.add_argument("--worst-day", action="store_true",
                    help="use the largest one-day move in the ten-year in the sample")
    args = ap.parse_args()

    curve0, curve1, dates = load_curves(args.date, args.worst_day)
    book = load_book()

    # Per-trade marks (day 0).
    rows = [{
        "trade": s.trade_id, "side": s.side, "notional": s.notional,
        "fixed": s.fixed_rate, "maturity": s.maturity,
        "npv_day0": swap_npv(curve0, s, 0.0), "dv01": dv01(curve0, s, 0.0),
    } for s in book]
    book_df = pd.DataFrame(rows)

    pnl = attribute(book, curve0, curve1, t0=0.0, t1=ONE_DAY)

    RESULTS.mkdir(exist_ok=True)
    book_df.to_csv(RESULTS / "book_marks.csv", index=False)
    pd.Series(pnl).to_csv(RESULTS / "pnl_attribution.csv")
    plot_curves(curve0, curve1, RESULTS / "sofr_curve.png", labels=dates)
    plot_waterfall(pnl, RESULTS / "pnl_waterfall.png")

    print(f"Book: {len(book)} swaps | day0 NPV {book_df['npv_day0'].sum():>14,.0f} | "
          f"net DV01 {book_df['dv01'].sum():>10,.0f}\n")
    print("Daily P&L attribution ({} -> {}):".format(*dates))
    for k in ("carry", "roll_down", "level", "slope", "curvature", "residual", "total"):
        print(f"  {k:<10} {pnl[k]:>14,.2f}")
    print(f"  {'(check)':<10} {pnl['check']:>14.2e}  (parts - total, should be ~0)")
    print(f"\nwrote -> {RESULTS}/  (book_marks.csv, pnl_attribution.csv, sofr_curve.png, pnl_waterfall.png)")


if __name__ == "__main__":
    main()
