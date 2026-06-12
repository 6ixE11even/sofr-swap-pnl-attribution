"""
Swap valuation.

Using the bond-minus-floater identity, a payer swap (pays fixed, receives float) is
worth, per unit notional, valued at a reset date t_v:

    NPV_payer = floatPV - fixedPV
              = (1 - DF(maturity))  -  fixed_rate * Σ τ_i DF(t_i)

The float leg is taken at par at the reset (its PV telescopes to 1 - DF(maturity)),
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


def swap_npv(curve: SofrCurve, swap: Swap, t_val: float = 0.0) -> float:
    """Mark-to-market of the swap at valuation time `t_val` (years)."""
    float_pv = 1.0 - float(curve.df(swap.maturity - t_val))
    fixed_pv = swap.fixed_rate * annuity(curve, swap, t_val)
    return swap.sign * swap.notional * (float_pv - fixed_pv)


def par_rate(curve: SofrCurve, swap: Swap, t_val: float = 0.0) -> float:
    """The fixed rate that makes the swap worth zero — i.e. the fair market rate."""
    float_pv = 1.0 - float(curve.df(swap.maturity - t_val))
    return float_pv / annuity(curve, swap, t_val)


def dv01(curve: SofrCurve, swap: Swap, t_val: float = 0.0, bump: float = 1e-4) -> float:
    """Dollar value of a 1bp parallel rise in zero rates (central difference)."""
    up = swap_npv(curve.shift(bump), swap, t_val)
    down = swap_npv(curve.shift(-bump), swap, t_val)
    return (up - down) / 2.0


def portfolio_npv(curve: SofrCurve, swaps: list[Swap], t_val: float = 0.0) -> float:
    return sum(swap_npv(curve, s, t_val) for s in swaps)
