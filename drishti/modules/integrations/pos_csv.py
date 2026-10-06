"""integration.pos_csv - replays bills exported from a real billing system.

Most small-store billing software can export a CSV. Point this at it to compute real
conversion rates. Columns expected (header row): timestamp (ISO 8601), bill_id, amount.

Config:
    path: data/bills.csv
    time_column: timestamp

Publishes: event/pos_bill   {bill_id, amount}

TODO(owner: integrations): a live connector (REST/webhook, or polling the POS database)
that publishes the same event while the store is open.
"""

from __future__ import annotations

import csv
import threading
from datetime import datetime

from drishti.core import Source


class PosCsv(Source):
    def setup(self) -> None:
        self.require("path")

    def run(self, stop: threading.Event) -> None:
        col = self.cfg.get("time_column", "timestamp")
        with open(self.cfg["path"], newline="") as f:
            for row in csv.DictReader(f):
                if stop.is_set():
                    break
                ts = datetime.fromisoformat(row[col]).timestamp()
                self.emit("pos_bill", "pos", ts, bill_id=row.get("bill_id", ""), amount=float(row.get("amount", 0) or 0))
