"""
Counterparty exposure and CVA on the swap book.

    uv run python scripts/run_cva.py
    uv run python scripts/run_cva.py --quality AAA --paths 20000

Exposure comes from block-bootstrapping real daily Treasury curve moves and real daily
credit-spread moves from the *same* historical days, so the correlation between the two
is whatever the market has actually shown rather than a number someone picked. Running
it again with the two resampled from different days is the wrong-way-risk experiment.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sofr_swap.curve import SofrCurve                              # noqa: E402
from sofr_swap.cva import (RECOVERY, cva, exposure, hazard_from_spread,  # noqa: E402
                           profile, simulate)
from sofr_swap.instruments import Swap                             # noqa: E402
from sofr_swap.market import latest_curve_inputs, market_panel     # noqa: E402
from sofr_swap.viz import plot_cva_decomposition, plot_exposure_profile  # noqa: E402


def load_book() -> list[Swap]:
    rows = pd.read_csv(ROOT / "data" / "positions.csv").to_dict("records")
    return [Swap(notional=float(r["notional"]), fixed_rate=float(r["fixed_rate"]),
                 effective=float(r["effective"]), maturity=float(r["maturity"]),
                 side=str(r["side"]), trade_id=str(r["trade_id"])) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quality", default="BAA", choices=["AAA", "BAA"],
                    help="counterparty credit quality (which FRED spread series)")
    ap.add_argument("--paths", type=int, default=20_000)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--block", type=int, default=10, help="bootstrap block length in days")
    ap.add_argument("--recovery", type=float, default=RECOVERY)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    panel = market_panel(args.quality)
    tenors, par, asof = latest_curve_inputs(panel)
    curve = SofrCurve.bootstrap(tenors, par)
    book = load_book()
    horizon = max(s.maturity for s in book)

    spot = float(panel["spread"].iloc[-1])
    print(f"curve as of {asof:%Y-%m-%d} from FRED CMT pillars "
          f"({', '.join(f'{t}y {r:.2%}' for t, r in zip(tenors, par))})")
    print(f"{len(panel):,} joint daily observations, "
          f"{panel.index.min():%Y-%m-%d} to {panel.index.max():%Y-%m-%d}")
    print(f"counterparty {args.quality}: spread {spot:.2%}, "
          f"hazard {hazard_from_spread(spot, args.recovery):.2%}/yr at R={args.recovery:.0%}\n")

    runs = {}
    for label, decouple in (("correlated (as observed)", False), ("independent", True)):
        sim = simulate(panel, horizon, args.steps, args.paths, block=args.block,
                       seed=args.seed, decouple=decouple)
        values = exposure(sim, book)
        prof = profile(values)
        result = cva(sim, values, curve, recovery=args.recovery, own_spread=spot / 2)
        runs[label] = {"sim": sim, "values": values, "profile": prof, **result}

    base = runs["correlated (as observed)"]
    sim = base["sim"]
    print(f"curve factors retained: {len(sim.variance_explained)} "
          f"({', '.join(f'{v:.1%}' for v in sim.variance_explained)}"
          f" = {sim.variance_explained.sum():.1%} of daily variance)")
    print(f"pillar draws pinned to the historical envelope: {sim.clipped:.1%}; "
          f"curves pulled back to a bootstrappable shape: {sim.repaired:.3%}\n")

    prof = base["profile"]
    peak = int(prof["epe"].idxmax())
    print(f"{'horizon':>9}{'EE':>14}{'EPE':>14}{'PFE 97.5%':>14}")
    for k in sorted({0, args.steps // 6, peak, args.steps // 2, args.steps - 1}):
        print(f"{sim.times[k]:>8.1f}y{prof.ee[k]:>14,.0f}{prof.epe[k]:>14,.0f}{prof.pfe[k]:>14,.0f}")
    print(f"\npeak EPE ${prof.epe[peak]:,.0f} at {sim.times[peak]:.1f}y; "
          f"peak PFE ${prof.pfe.max():,.0f} at {sim.times[int(prof.pfe.idxmax())]:.1f}y")

    print(f"\n{'':<26}{'CVA':>13}{'DVA':>13}{'BCVA':>13}")
    for label, r in runs.items():
        print(f"  {label:<24}{r['cva']:>13,.0f}{r['dva']:>13,.0f}{r['bcva']:>13,.0f}")
    wwr = base["cva"] - runs["independent"]["cva"]
    print(f"  {'wrong-way risk':<24}{wwr:>13,.0f}"
          f"   ({wwr / runs['independent']['cva']:+.1%} of the independent CVA)")

    # The mechanism, not just the number. The curve paths are identical across the two
    # runs by construction (same generator, same draws) so the difference above is the
    # credit coupling and not Monte Carlo noise - which is why DVA comes out equal.
    net = sum(s.sign * s.notional for s in book)
    rate_credit = np.corrcoef(panel.diff().dropna()[10], panel.diff().dropna()["spread"])[0, 1]
    # Correlate within each time slice, not pooled: exposure and the spread both drift
    # with the horizon, and pooling lets the two trends cancel to a flat zero. Quote it
    # where the CVA is actually earned, at the peak of the exposure profile - by thirty
    # years only two trades are left alive and the correlation there prices nothing.
    ex_credit = float(np.corrcoef(base["values"][:, peak], sim.spreads[:, peak])[0, 1])
    print(f"\n  book is net {'payer' if net > 0 else 'receiver'} ${abs(net) / 1e6:,.0f}mm; "
          f"corr(d 10y, d spread) = {rate_credit:+.2f} in the data")
    print(f"  so exposure rises as the counterparty deteriorates: "
          f"corr(book MTM, spread) = {ex_credit:+.2f} at the {sim.times[peak]:.1f}y "
          f"exposure peak")

    rows = []
    for label, r in runs.items():
        rows.append({"case": label, "cva": r["cva"], "dva": r["dva"], "bcva": r["bcva"],
                     "peak_epe": float(r["profile"].epe.max()),
                     "peak_pfe": float(r["profile"].pfe.max())})
    out = ROOT / "reports"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "cva_summary.csv", index=False)
    base["profile"].assign(time=sim.times).to_csv(out / "exposure_profile.csv", index=False)
    plot_exposure_profile(sim.times, base["profile"], out / "figures" / "exposure_profile.png")
    plot_cva_decomposition(runs, out / "figures" / "cva_wrong_way.png")
    print(f"\nwrote -> {out}/ (cva_summary.csv, exposure_profile.csv, figures/)")


if __name__ == "__main__":
    main()
