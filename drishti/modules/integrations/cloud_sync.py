"""integration.cloud_sync - pushes small daily summaries to a head-office server when online.

Offline-first: the store box never waits for the internet. Every `interval_s` it tries to
upload a JSON summary of events since the last successful upload; if the network is down it
simply tries again later. Only counts and alerts leave the store - never images.

Config:
    endpoint: https://example.org/api/stores/demo/summary
    token: ""                  sent as a Bearer token
    interval_s: 900
    db: events.db              the event store file (relative to out_dir)

TODO(owner: integrations): not run in CI. Decide the summary format with the dashboard owner
(multi-store view) and add retries with backoff.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import urllib.request
from pathlib import Path

from drishti.core import Source


class CloudSync(Source):
    def setup(self) -> None:
        self.require("endpoint")
        db = Path(self.cfg.get("db", "events.db"))
        self.db = db if db.is_absolute() else self.ctx.out_dir / db
        self.watermark = 0.0

    def summary(self, since: float) -> dict:
        con = sqlite3.connect(str(self.db))
        rows = con.execute("SELECT type, COUNT(*) FROM events WHERE ts > ? GROUP BY type", (since,)).fetchall()
        alerts = con.execute("SELECT ts, kind, message FROM alerts WHERE ts > ? ORDER BY ts", (since,)).fetchall()
        last = con.execute("SELECT MAX(ts) FROM events").fetchone()[0] or since
        con.close()
        return {"since": since, "until": last, "event_counts": dict(rows),
                "alerts": [{"ts": a[0], "kind": a[1], "message": a[2]} for a in alerts]}

    def run(self, stop: threading.Event) -> None:
        interval = float(self.cfg.get("interval_s", 900))
        while not stop.wait(interval):
            if not self.db.exists():
                continue
            body = self.summary(self.watermark)
            req = urllib.request.Request(self.cfg["endpoint"], data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {self.cfg.get('token', '')}"})
            try:
                with urllib.request.urlopen(req, timeout=10):
                    self.watermark = body["until"]
                    self.log.info("synced summary up to %s", time.ctime(self.watermark))
            except OSError as e:
                self.log.info("offline, will retry: %s", e)
