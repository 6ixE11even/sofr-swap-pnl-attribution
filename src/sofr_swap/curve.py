"""
The SOFR discount curve.

Single-curve world: SOFR is used for both projection and discounting (standard for
a collateralised USD swap book). The curve is a set of pillar discount factors with
log-linear interpolation — i.e. piecewise-constant instantaneous forward rates,
which keeps forwards sane between pillars.
"""
from __future__ import annotations

import numpy as np

from sofr_swap.conventions import TAU


class SofrCurve:
    def __init__(self, times, dfs):
        """`times` in years from the curve's anchor date (must start at 0 with df=1)."""
        self.times = np.asarray(times, dtype=float)
        self._log_df = np.log(np.asarray(dfs, dtype=float))

    def df(self, t):
        """Discount factor at time t (years). Log-linear in t."""
        return np.exp(np.interp(np.asarray(t, dtype=float), self.times, self._log_df))

    def zero_rate(self, t):
        """Continuously-compounded zero rate, z(t) = -ln df(t) / t."""
        t = np.asarray(t, dtype=float)
        return np.where(t > 0, -np.log(self.df(t)) / np.where(t > 0, t, 1.0), 0.0)

    def forward_rate(self, t1: float, t2: float) -> float:
        """Simple forward rate over [t1, t2]."""
        return (self.df(t1) / self.df(t2) - 1.0) / (t2 - t1)

    # --- pillar access / shifting (for sensitivities and scenarios) -----------
    def pillar_times(self) -> np.ndarray:
        return self.times[1:]  # drop the t=0 anchor

    def pillar_zeros(self) -> np.ndarray:
        t = self.times[1:]
        return -self._log_df[1:] / t

    def shift(self, dz) -> "SofrCurve":
        """Return a new curve with pillar zero rates shifted by `dz` (scalar or
        per-pillar array). The workhorse for DV01, key-rate durations, scenarios."""
        t = self.times[1:]
        z = self.pillar_zeros() + dz
        return SofrCurve(self.times, np.concatenate([[1.0], np.exp(-z * t)]))

    # --- construction ---------------------------------------------------------
    @classmethod
    def bootstrap(cls, tenors, par_rates) -> "SofrCurve":
        """Bootstrap discount factors from par swap rates.

        Par rates are first linearly interpolated onto an annual pillar grid so the
        bootstrap is a clean forward substitution: for each maturity k,
            DF_k = (1 - S_k * A_{k-1}) / (1 + S_k * τ),   A = running annuity.
        """
        tenors = np.asarray(tenors, dtype=float)
        par_rates = np.asarray(par_rates, dtype=float)
        grid = np.arange(1, int(tenors.max()) + 1)
        par = np.interp(grid, tenors, par_rates)

        dfs, annuity = [], 0.0
        for s in par:
            df_k = (1.0 - s * annuity) / (1.0 + s * TAU)
            dfs.append(df_k)
            annuity += TAU * df_k

        return cls(np.concatenate([[0.0], grid]), np.concatenate([[1.0], dfs]))
