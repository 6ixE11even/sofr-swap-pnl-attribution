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


def test_forward_starting_swap_prices_off_its_own_start():
    """The float leg starts at `effective`, not at the valuation date.

    Pricing a 5y5y as if it were already accruing put its par rate near 8.8%
    instead of the ~4% the curve implies.
    """
    c = _curve()
    fwd = Swap(1e7, 0.0, 5.0, 10.0, side="payer", trade_id="5y5y")
    s = par_rate(c, fwd)
    assert 0.030 < s < 0.050
    assert abs(swap_npv(c, Swap(1e7, s, 5.0, 10.0, side="payer"))) < 1e-6


def test_curve_extrapolates_flat_forward_not_flat_discount():
    """Past the last pillar numpy's clamp held DF constant — a zero forward rate."""
    c = _curve()
    last = max(TENORS)
    assert c.df(last + 5.0) < c.df(last)
    f_in = c.forward_rate(last - 5.0, last)
    f_out = c.forward_rate(last, last + 5.0)
    assert abs(f_out - f_in) < 5e-3          # forward carries on, roughly flat
    assert abs(c.zero_rate(last + 5.0) - c.zero_rate(last)) < 2e-3


def test_par_rate_on_a_dead_swap_names_the_trade():
    c = _curve()
    try:
        par_rate(c, Swap(1e7, 0.04, 0.0, 0.5, trade_id="SW99"))
    except ValueError as e:
        assert "SW99" in str(e)
    else:
        raise AssertionError("expected a ValueError, got a bare ZeroDivisionError")


def test_carry_skips_swaps_that_are_not_accruing():
    """A forward-starting trade pays no coupon today."""
    from sofr_swap.attribution import _carry
    c = _curve()
    live = Swap(1e7, 0.04, 0.0, 10.0, "payer")
    unstarted = Swap(1e7, 0.04, 5.0, 10.0, "payer")
    assert _carry([live, unstarted], c, 0.0, 1 / 365) == _carry([live], c, 0.0, 1 / 365)


# --- counterparty exposure and CVA ------------------------------------------

import pytest  # noqa: E402

from sofr_swap.cva import (_interp_matrix, bootstrap_dfs, cva, exposure,  # noqa: E402
                           hazard_from_spread, profile, simulate)
from sofr_swap.market import market_panel  # noqa: E402
from sofr_swap.pricing import portfolio_npv  # noqa: E402

PANEL = market_panel("BAA")
PILLARS = [c for c in PANEL.columns if c != "spread"]
BOOK = [Swap(notional=25e6, fixed_rate=0.041, effective=0, maturity=2, side="payer", trade_id="A"),
        Swap(notional=40e6, fixed_rate=0.0395, effective=0, maturity=10, side="receiver", trade_id="B"),
        Swap(notional=18e6, fixed_rate=0.0402, effective=0, maturity=5, side="receiver", trade_id="C")]


def _sim(**kw):
    return simulate(PANEL, horizon=10, steps=20, n_paths=400, seed=0, **kw)


def test_credit_triangle_round_trips():
    """lambda = s/(1-R) has to reprice to the spread it came from."""
    for spread in (0.0050, 0.0158, 0.0400):
        for recovery in (0.0, 0.4, 0.7):
            lam = hazard_from_spread(spread, recovery)
            assert lam * (1 - recovery) == pytest.approx(spread)


def test_vectorised_bootstrap_agrees_with_the_single_curve_one():
    """The fast path exists for speed, not for a different answer."""
    par = np.interp(np.arange(1, 31), [float(t) for t in PILLARS],
                    PANEL[PILLARS].iloc[-1].to_numpy())
    fast = bootstrap_dfs(par[None, :], 30)[0]
    slow = SofrCurve.bootstrap(np.arange(1, 31), par)
    assert np.allclose(fast, slow.df(np.arange(1, 31)), rtol=1e-12)


def test_interpolation_matrix_reproduces_the_curve_object():
    """Including past the last pillar, where the curve carries the forward flat."""
    par = np.interp(np.arange(1, 31), [float(t) for t in PILLARS],
                    PANEL[PILLARS].iloc[-1].to_numpy())
    curve = SofrCurve.bootstrap(np.arange(1, 31), par)
    grid = np.arange(1, 31, dtype=float)
    targets = np.array([0.4, 1.0, 3.7, 12.5, 29.0, 30.0, 34.5])
    got = np.exp(np.log(bootstrap_dfs(par[None, :], 30)) @ _interp_matrix(targets, grid).T)[0]
    assert np.allclose(got, curve.df(targets), rtol=1e-10)


def test_exposure_at_the_first_node_is_a_swap_valuation():
    """The exposure engine must price the book the way pricing.py does."""
    sim = _sim()
    values = exposure(sim, BOOK)
    t = float(sim.times[0])
    for i in range(5):
        curve = SofrCurve.bootstrap(sim.tenor_grid, sim.par_rates[i, 0, :])
        assert values[i, 0] == pytest.approx(portfolio_npv(curve, BOOK, t), rel=1e-9)


def test_every_simulated_curve_has_positive_discount_factors():
    """A curve whose bootstrap goes negative is not a curve, and it would poison
    every exposure downstream of it as a NaN."""
    sim = _sim()
    for k in range(sim.times.size):
        assert (bootstrap_dfs(sim.par_rates[:, k, :], sim.tenor_grid.size) > 0).all()


def test_the_bootstrap_carries_no_drift():
    """Forty years of Treasury history has a large downward trend in it. Resampling it
    without demeaning walks the ten-year somewhere absurd over a long horizon."""
    sim = simulate(PANEL, horizon=30, steps=30, n_paths=3_000, seed=1)
    start = float(PANEL[10].iloc[-1])
    drift = sim.par_rates[:, -1, 9].mean() - start
    assert abs(drift) < 0.005, f"mean 10y drifted {1e4 * drift:.0f}bp over thirty years"


def test_a_counterparty_that_cannot_default_costs_nothing():
    sim = _sim()
    sim.spreads[:] = 0.0
    values = exposure(sim, BOOK)
    curve = SofrCurve.bootstrap(PANEL[PILLARS].columns.astype(float),
                                PANEL[PILLARS].iloc[-1].to_numpy())
    assert cva(sim, values, curve)["cva"] == pytest.approx(0.0, abs=1e-9)


def test_cva_rises_with_the_spread_and_falls_with_recovery():
    sim = _sim()
    values = exposure(sim, BOOK)
    curve = SofrCurve.bootstrap(PANEL[PILLARS].columns.astype(float),
                                PANEL[PILLARS].iloc[-1].to_numpy())
    cheap = cva(sim, values, curve)["cva"]
    sim.spreads *= 2.0
    dear = cva(sim, values, curve)["cva"]
    assert dear > cheap
    assert cva(sim, values, curve, recovery=0.8)["cva"] < dear


def test_decoupling_changes_only_the_credit_leg():
    """The wrong-way experiment is only clean if the curve paths are bit-identical:
    otherwise the difference in CVA is partly Monte Carlo noise."""
    joint, split = _sim(), _sim(decouple=True)
    assert np.array_equal(joint.par_rates, split.par_rates)
    assert not np.array_equal(joint.spreads, split.spreads)
    assert np.array_equal(exposure(joint, BOOK), exposure(split, BOOK))


def test_exposure_profile_orders_the_way_the_definitions_require():
    sim = _sim()
    prof = profile(exposure(sim, BOOK))
    assert (prof["pfe"] >= prof["epe"] - 1e-6).all()
    assert (prof["epe"] >= prof["ee"] - 1e-6).all()
    assert (prof["ee"] >= prof["ene"] - 1e-6).all()
    assert (prof["ene"] <= 1e-6).all()
