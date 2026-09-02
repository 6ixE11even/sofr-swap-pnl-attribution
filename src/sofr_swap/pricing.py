"""
Swap valuation.

Using the bond-minus-floater identity, a payer swap (pays fixed, receives float) is
worth, per unit notional, valued at a reset date t_v:

    NPV_payer = floatPV - fixedPV
              = (DF(start) - DF(maturity))  -  fixed_rate * Σ τ_i DF(t_i)

`start` is the accrual start seen from t_v: zero once the swap has begun, so the
float leg collapses to the familiar 1 - DF(maturity). For a forward-starting swap
the DF(start) term matters — dropping it prices the unstarted stub as if it were
already accruing, which throws the par rate out by hundreds of basis points.

The float leg is taken at par at the reset (its PV telescopes to DF(start) - DF(maturity)),
which is exact on reset dates and a small, consistent approximation between them —
fine for a daily P&L, and the assumption is stated in the README.
"""
from __future__ import annotations

from sofr_swap.conventions import TAU
from sofr_swap.curve import SofrCurve
from sofr_swap.instruments import Swap


def annuity(curve: SofrCurve, swap: Swap, t_val: float) -> float:
    """PV01 of the fixed leg per unit notional: Σ τ_i DF(t_i) over future coupons."""
    total = 0.0
    for _start, end in swap.schedule():
        if end > t_val + 1e-9:                       # only coupons that haven't paid
            total += TAU * float(curve.df(end - t_val))
    return total


def float_leg_pv(curve: SofrCurve, swap: Swap, t_val: float = 0.0) -> float:
    """PV of the floating leg per unit notional: DF(start) - DF(maturity)."""
    if swap.maturity <= t_val + 1e-9:
        return 0.0
    start = max(swap.effective - t_val, 0.0)
    return float(curve.df(start)) - float(curve.df(swap.maturity - t_val))


def swap_npv(curve: SofrCurve, swap: Swap, t_val: float = 0.0) -> float:
    """Mark-to-market of the swap at valuation time `t_val` (years)."""
    float_pv = float_leg_pv(curve, swap, t_val)
    fixed_pv = swap.fixed_rate * annuity(curve, swap, t_val)
    return swap.sign * swap.notional * (float_pv - fixed_pv)


def par_rate(curve: SofrCurve, swap: Swap, t_val: float = 0.0) -> float:
    """The fixed rate that makes the swap worth zero — i.e. the fair market rate."""
    ann = annuity(curve, swap, t_val)
    if ann <= 0.0:
        raise ValueError(
            f"swap {swap.trade_id or '<unnamed>'} has no coupons left after t_val={t_val}"
            f" (maturity {swap.maturity}); par rate is undefined"
        )
    return float_leg_pv(curve, swap, t_val) / ann


def dv01(curve: SofrCurve, swap: Swap, t_val: float = 0.0, bump: float = 1e-4) -> float:
    """Dollar value of a 1bp parallel rise in zero rates (central difference)."""
    up = swap_npv(curve.shift(bump), swap, t_val)
    down = swap_npv(curve.shift(-bump), swap, t_val)
    return (up - down) / 2.0


def portfolio_npv(curve: SofrCurve, swaps: list[Swap], t_val: float = 0.0) -> float:
    return sum(swap_npv(curve, s, t_val) for s in swaps)
