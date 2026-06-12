"""Vanilla fixed-vs-SOFR interest-rate swap."""
from __future__ import annotations

from dataclasses import dataclass

from sofr_swap.conventions import annual_schedule


@dataclass
class Swap:
    notional: float          # in currency units (e.g. 10_000_000 = $10mm)
    fixed_rate: float        # the contractual fixed coupon (decimal, e.g. 0.042)
    effective: float         # start, in years from epoch
    maturity: float          # end, in years from epoch
    side: str = "payer"      # "payer" pays fixed / receives float; "receiver" the reverse
    trade_id: str = ""

    def schedule(self) -> list[tuple[float, float]]:
        return annual_schedule(self.effective, self.maturity)

    @property
    def sign(self) -> int:
        return 1 if self.side == "payer" else -1
