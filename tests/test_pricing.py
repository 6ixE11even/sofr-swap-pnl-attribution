"""Sanity checks on the curve, pricer, and attribution."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sofr_swap.attribution import attribute
from sofr_swap.curve import SofrCurve
from sofr_swap.instruments import Swap
from sofr_swap.pricing import dv01, par_rate, swap_npv

TENORS = [1, 2, 3, 5, 7, 10, 15, 20, 30]
RATES = [0.0455, 0.0420, 0.0400, 0.0388, 0.0390, 0.0398, 0.0410, 0.0415, 0.0405]


def _curve(bump=0.0):
    return SofrCurve.bootstrap(TENORS, [r + bump for r in RATES])


def test_bootstrap_reprices_par_swaps():
    c = _curve()
    for T, S in zip(TENORS, RATES):
        sw = Swap(1.0, S, 0.0, float(T))
        assert abs(swap_npv(c, sw)) < 1e-9        # par swap is worth zero
        assert abs(par_rate(c, sw) - S) < 1e-9    # model par == input par


def test_dv01_signs():
    c = _curve()
    payer = Swap(1e7, 0.04, 0.0, 10.0, side="payer")
    receiver = Swap(1e7, 0.04, 0.0, 10.0, side="receiver")
    assert dv01(c, payer) > 0       # payer gains when rates rise
    assert dv01(c, receiver) < 0
    assert abs(dv01(c, payer) + dv01(c, receiver)) < 1e-6  # equal and opposite


def test_off_market_swap_sign():
    c = _curve()
    # paying 3% fixed when the market is ~4% is an asset to the payer
    assert swap_npv(c, Swap(1e7, 0.03, 0.0, 10.0, side="payer")) > 0


def test_attribution_is_additive():
    swaps = [Swap(1e7, 0.04, 0.0, 10.0, "payer"), Swap(2e7, 0.039, 0.0, 5.0, "receiver")]
    c0, c1 = _curve(), _curve(bump=0.0005)  # +5bp parallel
    pnl = attribute(swaps, c0, c1, t0=0.0, t1=1 / 365)
    parts = sum(pnl[k] for k in ("carry", "roll_down", "level", "slope", "curvature", "residual"))
    assert abs(parts - pnl["total"]) < 1e-6
    # a clean +5bp parallel move should load almost entirely on "level"
    assert abs(pnl["level"]) > abs(pnl["slope"])
    assert np.isfinite(pnl["total"])
