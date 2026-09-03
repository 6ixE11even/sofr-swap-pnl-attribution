"""
Real market data from FRED, no API key required.

Two families, both keyless CSV:

- Treasury constant-maturity par yields (DGS1 ... DGS30) — the pillar grid the
  discount curve is bootstrapped from, daily back to the 1960s-80s depending on tenor.
- Moody's corporate spreads over the 10-year (AAA10Y, BAA10Y) — daily since 1983 and
  1986. These are used instead of the ICE BofA OAS series (BAMLC0A4CBBB and friends)
  for one reason: FRED's public CSV endpoint truncates ICE BofA data to the last three
  years whatever `cosd` you pass, and three years of credit history contains no crisis.
  Moody's comes back whole, and 1998, 2008, 2011 and 2020 are all in it.

Everything is cached under `data/fred/` so a rerun is offline and so every number in
the README corresponds to a file on disk.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
CACHE = Path(__file__).resolve().parents[2] / "data" / "fred"

# Treasury CMT pillars: series id -> tenor in years.
PILLARS = {"DGS1": 1, "DGS2": 2, "DGS3": 3, "DGS5": 5,
           "DGS7": 7, "DGS10": 10, "DGS20": 20, "DGS30": 30}

# Counterparty credit quality -> the spread series that stands in for it.
CREDIT = {"AAA": "AAA10Y", "BAA": "BAA10Y"}


def fred_series(series: str, refresh: bool = False) -> pd.Series:
    """One FRED series as a float Series indexed by date, cached to disk."""
    path = CACHE / f"{series}.csv"
    if path.exists() and not refresh:
        frame = pd.read_csv(path)
    else:
        frame = pd.read_csv(FRED.format(series=series))
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    frame.columns = ["date", series]
    frame["date"] = pd.to_datetime(frame["date"])
    # FRED writes "." for a market holiday. to_numeric turns those into NaN; dropping
    # them is right, filling them forward would invent a day the market did not trade
    # and put a zero in the middle of the daily-change distribution.
    values = pd.to_numeric(frame[series], errors="coerce")
    return pd.Series(values.to_numpy(), index=frame["date"], name=series).dropna()


def rates_panel(refresh: bool = False) -> pd.DataFrame:
    """Par yields at every pillar, in decimals. No credit series, so the history is not
    shortened to the shorter of the two - useful when only the curve is wanted."""
    columns = {tenor: fred_series(s, refresh) for s, tenor in PILLARS.items()}
    return pd.concat(columns, axis=1, join="inner").sort_index() / 100.0


def consecutive_pairs(panel: pd.DataFrame, max_gap_days: int = 5) -> pd.Series:
    """Boolean mask of rows whose predecessor is a genuine previous business day.

    The pillars have holes - the thirty-year was not issued between 2002 and 2006, and
    an inner join simply drops those years. Two adjacent *rows* of this frame can then
    be seven years apart, and calling their difference a one-day move produces an
    eight-million-dollar carry number with a straight face.
    """
    gap = panel.index.to_series().diff().dt.days
    return (gap <= max_gap_days).fillna(False)


def market_panel(quality: str = "BAA", refresh: bool = False) -> pd.DataFrame:
    """Par yields at every pillar plus the counterparty's credit spread, in decimals.

    Inner-joined on date: a day where any pillar is missing is a day the whole curve
    cannot be built, and a daily change spanning that gap is a two-day move being
    counted as one.
    """
    columns = {tenor: fred_series(s, refresh) for s, tenor in PILLARS.items()}
    columns["spread"] = fred_series(CREDIT[quality], refresh)
    panel = pd.concat(columns, axis=1, join="inner").sort_index()
    return panel / 100.0                    # FRED quotes percent; the curve wants decimals


def latest_curve_inputs(panel: pd.DataFrame) -> tuple[list[float], list[float], pd.Timestamp]:
    """The most recent complete row: (tenors, par rates, date)."""
    row = panel.iloc[-1]
    tenors = [c for c in panel.columns if c != "spread"]
    return tenors, [float(row[t]) for t in tenors], panel.index[-1]
