"""output.console - prints alerts (and optionally chosen events) to the terminal.

Config:
    events: [entry, queue_forecast]   also print these event types (default: none)
    tz: "+05:30"                      store time zone for printed timestamps

Listens: alert/#, event/#
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from drishti.core import Alert, Event, Module

ICON = {"info": "i", "warn": "!", "critical": "!!"}


def parse_tz(s: str) -> timezone:
    sign = -1 if s.startswith("-") else 1
    hh, mm = s.lstrip("+-").split(":")
    return timezone(sign * timedelta(hours=int(hh), minutes=int(mm)))


class ConsoleNotifier(Module):
    subscribes = ("alert/#", "event/#")

    def setup(self) -> None:
        self.events = set(self.cfg.get("events", []))
        self.tz = parse_tz(self.cfg.get("tz", self.ctx.store.get("tz", "+05:30")))

    def _t(self, ts: float) -> str:
        return datetime.fromtimestamp(ts, self.tz).strftime("%H:%M:%S")

    def on_message(self, topic: str, msg: Any) -> None:
        if isinstance(msg, Alert):
            print(f"[{self._t(msg.ts)}] ALERT {ICON.get(msg.severity, '')} {msg.kind}: {msg.message}", flush=True)
        elif isinstance(msg, Event) and msg.type in self.events:
            shown = {k: v for k, v in msg.data.items() if k != "grid"}
            print(f"[{self._t(msg.ts)}] {msg.type} {msg.camera_id} {shown}", flush=True)
