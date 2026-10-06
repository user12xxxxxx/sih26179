"""output.event_store - append-only SQLite log of every event, alert and metric.

This file is the contract between the live pipeline and everything else: the dashboard,
reports and cloud sync only read it. A recorded events.db from the simulator lets the UI
team work without any camera or model running.

Config:
    path: events.db           relative to out_dir
    commit_every: 200         rows per transaction (always committed on shutdown)

Listens: event/#, alert/#, metric/#

Schema:
    events (id, ts, type, camera_id, source, data JSON)
    alerts (id, ts, kind, severity, message, camera_id, source, data JSON)
    metrics(id, ts, name, camera_id, source, value)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from drishti.core import Alert, Event, Metric, Module

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, ts REAL, type TEXT, camera_id TEXT, source TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY, ts REAL, kind TEXT, severity TEXT, message TEXT,
                                   camera_id TEXT, source TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS metrics (id INTEGER PRIMARY KEY, ts REAL, name TEXT, camera_id TEXT, source TEXT, value REAL);
CREATE INDEX IF NOT EXISTS events_type_ts ON events(type, ts);
CREATE INDEX IF NOT EXISTS alerts_ts ON alerts(ts);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path), check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")  # readers (dashboard) never block the writer
    con.executescript(SCHEMA)
    return con


class EventStore(Module):
    subscribes = ("event/#", "alert/#", "metric/#")

    def setup(self) -> None:
        path = Path(self.cfg.get("path", "events.db"))
        if not path.is_absolute():
            path = self.ctx.out_dir / path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.con = connect(path)
        self.every = int(self.cfg.get("commit_every", 200))
        self.pending = 0

    def on_message(self, topic: str, msg: Any) -> None:
        if isinstance(msg, Event):
            self.con.execute("INSERT INTO events(ts,type,camera_id,source,data) VALUES (?,?,?,?,?)",
                             (msg.ts, msg.type, msg.camera_id, msg.source, json.dumps(msg.data)))
        elif isinstance(msg, Alert):
            self.con.execute("INSERT INTO alerts(ts,kind,severity,message,camera_id,source,data) VALUES (?,?,?,?,?,?,?)",
                             (msg.ts, msg.kind, msg.severity, msg.message, msg.camera_id, msg.source, json.dumps(msg.data)))
        elif isinstance(msg, Metric):
            self.con.execute("INSERT INTO metrics(ts,name,camera_id,source,value) VALUES (?,?,?,?,?)",
                             (msg.ts, msg.name, msg.camera_id, msg.source, msg.value))
        else:
            return
        self.pending += 1
        if self.pending >= self.every:
            self.con.commit()
            self.pending = 0

    def teardown(self) -> None:
        self.con.commit()
        self.con.close()
        self.log.info("event store written to %s", self.path)
