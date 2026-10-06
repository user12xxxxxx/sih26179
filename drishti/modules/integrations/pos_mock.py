"""integration.pos_mock - fake billing system: every finished checkout produces a bill.

Stands in for a real POS so conversion rate (bills / visitors) and "phantom stock" checks can
be built now. Replace with integration.pos_csv (or a real API connector) later - the
`event/pos_bill` contract stays the same.

Config:
    bill_prob: 0.97          share of served customers who actually pay
    amount_inr: [60, 1500]   uniform range
    seed: 11

Listens:   event/service_complete
Publishes: event/pos_bill   {bill_id, amount, counter}
"""

from __future__ import annotations

import random
from typing import Any

from drishti.core import Module


class PosMock(Module):
    subscribes = ("event/service_complete",)

    def setup(self) -> None:
        self.rng = random.Random(self.cfg.get("seed", 11))
        self.p = float(self.cfg.get("bill_prob", 0.97))
        self.lo, self.hi = self.cfg.get("amount_inr", [60, 1500])
        self.n = 0

    def on_message(self, topic: str, ev: Any) -> None:
        if self.rng.random() > self.p:
            return
        self.n += 1
        self.emit("pos_bill", "pos", ev.ts, bill_id=f"B{self.n:05d}", amount=round(self.rng.uniform(self.lo, self.hi), 2),
                  counter=ev.data.get("counter"))
