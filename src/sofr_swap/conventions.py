"""Day-count and schedule helpers.

Time is measured in years from a fixed epoch (the trade date). It keeps the maths
readable; a production system would carry real calendar dates and a holiday
calendar. USD SOFR swaps use ACT/360 and (for standard tenors) annual coupons,
which is what's modelled here.
"""
from __future__ import annotations

ACT_BASIS = 360.0
DAYS_PER_YEAR = 365.0

# Annual ACT/360 accrual fraction (one 365-day year over a 360 basis).
TAU = DAYS_PER_YEAR / ACT_BASIS


def year_fraction(t0: float, t1: float) -> float:
    """ACT/360 year fraction between two times given in years."""
    return (t1 - t0) * DAYS_PER_YEAR / ACT_BASIS


def annual_schedule(effective: float, maturity: float) -> list[tuple[float, float]]:
    """Annual (accrual_start, accrual_end) periods from `effective` to `maturity`."""
    periods, start, k = [], effective, 1
    while effective + k <= maturity + 1e-9:
        end = effective + k
        periods.append((start, end))
        start, k = end, k + 1
    return periods
