"""
End-to-end demo:  python scripts/run_attribution.py

Reads two days of SOFR par-swap quotes and a swap book, bootstraps a curve for
each day, marks the book, and attributes the one-day P&L into carry, roll-down,
and curve moves. Writes a table + two charts to results/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sofr_swap.attribution import attribute            # noqa: E402
from sofr_swap.curve import SofrCurve                  # noqa: E402
from sofr_swap.instruments import Swap                 # noqa: E402
from sofr_swap.pricing import dv01, swap_npv           # noqa: E402
from sofr_swap.viz import plot_curves, plot_waterfall  # noqa: E402

ONE_DAY = 1.0 / 365.0
RESULTS = ROOT / "results"


def load_curves() -> tuple[SofrCurve, SofrCurve, list[str]]:
    q = pd.read_csv(ROOT / "data" / "market_quotes.csv")
    dates = sorted(q["date"].unique())
    curves = []
    for d in dates[:2]:
        day = q[q["date"] == d].sort_values("tenor")
        curves.append(SofrCurve.bootstrap(day["tenor"].to_numpy(), day["par_rate"].to_numpy()))
    return curves[0], curves[1], dates[:2]


def load_book() -> list[Swap]:
    p = pd.read_csv(ROOT / "data" / "positions.csv")
    return [Swap(r.notional, r.fixed_rate, r.effective, r.maturity, r.side, r.trade_id)
            for r in p.itertuples()]


def main() -> None:
    curve0, curve1, dates = load_curves()
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
