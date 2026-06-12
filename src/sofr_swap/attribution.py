"""
Daily P&L attribution.

Between valuation dates t0 and t1 the change in portfolio mark-to-market is split
into pieces a desk actually reports:

    total = carry + roll-down + level + slope + curvature + residual

How:
  * Roll the SAME curve forward to t1     -> theta (time effect)
        carry      = net coupon accrued over the day (float - fixed)
        roll-down  = theta - carry  (sliding along a static curve)
  * Re-price with the NEW curve at t1      -> curve P&L (rate move)
        linearised via key-rate durations, then the pillar-rate change is projected
        onto level / slope / curvature shapes; whatever the linear + basis fit
        misses lands in `residual` (second-order convexity + off-shape moves).

The split is constructed to be exactly additive — the parts sum to `total`.
"""
from __future__ import annotations

import numpy as np

from sofr_swap.conventions import year_fraction
from sofr_swap.curve import SofrCurve
from sofr_swap.instruments import Swap
from sofr_swap.pricing import portfolio_npv


def _carry(swaps: list[Swap], curve0: SofrCurve, t0: float, t1: float) -> float:
    """Net coupon earned over [t0, t1]: float (≈ overnight SOFR) minus fixed."""
    dt = year_fraction(t0, t1)
    overnight = float(curve0.forward_rate(0.0, max(t1 - t0, 1.0 / 365.0)))
    return sum(s.sign * s.notional * (overnight - s.fixed_rate) * dt for s in swaps)


def attribute(swaps: list[Swap], curve0: SofrCurve, curve1: SofrCurve,
              t0: float, t1: float, bump: float = 1e-4) -> dict[str, float]:
    v0 = portfolio_npv(curve0, swaps, t0)
    v_static = portfolio_npv(curve0, swaps, t1)   # same curve, one day older
    v1 = portfolio_npv(curve1, swaps, t1)

    carry = _carry(swaps, curve0, t0, t1)
    rolldown = (v_static - v0) - carry
    curve_pnl = v1 - v_static

    # Key-rate durations of the portfolio (bump each pillar zero, reprice at t1).
    dz = curve1.pillar_zeros() - curve0.pillar_zeros()
    n = len(dz)
    krd = np.empty(n)
    for j in range(n):
        e = np.zeros(n)
        e[j] = bump
        krd[j] = (portfolio_npv(curve0.shift(e), swaps, t1) - v_static) / bump

    # Project the pillar-rate change onto level / slope / curvature.
    level = np.ones(n)
    slope = np.linspace(-1.0, 1.0, n)
    curv = slope ** 2 - np.mean(slope ** 2)
    basis = np.vstack([level, slope, curv]).T
    coef, *_ = np.linalg.lstsq(basis, dz, rcond=None)
    resid_z = dz - basis @ coef

    level_pnl = float(krd @ (coef[0] * level))
    slope_pnl = float(krd @ (coef[1] * slope))
    curv_pnl = float(krd @ (coef[2] * curv))
    # residual = basis misfit (linear) + convexity (curve_pnl beyond the linear KRD term)
    residual = float(krd @ resid_z) + (curve_pnl - float(krd @ dz))

    out = {
        "carry": carry,
        "roll_down": rolldown,
        "level": level_pnl,
        "slope": slope_pnl,
        "curvature": curv_pnl,
        "residual": residual,
        "total": v1 - v0,
    }
    out["check"] = sum(out[k] for k in ("carry", "roll_down", "level", "slope", "curvature", "residual")) - out["total"]
    return out
