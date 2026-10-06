"""output.alert_engine - YAML rules that turn events into alerts for staff.

Rules live in config so the shop owner's needs change without code changes:

    rules:
      - event: shelf_gap                   event type ("on:" would be read as a YAML boolean)
        where: {state: EMPTY}              all conditions must hold
        kind: restock
        severity: warn                     info | warn | critical
        message: "Refill {product} (slot {slot})"
        key: "{slot}"                      alerts with the same key are de-duplicated...
        cooldown_s: 600                    ...for this many seconds
      - event: queue_forecast
        where: {recommended_counters: {gt: $open_counters}}
        kind: open_counter
        severity: critical
        message: "Open counter {recommended_counters}: about {predicted_wait_min} min wait expected"
        key: queue
        cooldown_s: 300

Conditions: a plain value means "equals"; or use {eq|ne|gt|gte|lt|lte|in: value}.
A value written as "$field" refers to another field of the same event.

Listens:   event/#
Publishes: alert/<kind>
"""

from __future__ import annotations

import operator
import string
from typing import Any

from drishti.core import Alert, Event, Module

OPS = {"eq": operator.eq, "ne": operator.ne, "gt": operator.gt, "gte": operator.ge, "lt": operator.lt, "lte": operator.le,
       "in": lambda a, b: a in b}


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "?"


def _resolve(value: Any, data: dict[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        return data.get(value[1:])
    return value


def matches(where: dict[str, Any] | None, data: dict[str, Any]) -> bool:
    for field, cond in (where or {}).items():
        actual = data.get(field)
        if isinstance(cond, dict):
            for op, raw in cond.items():
                expected = _resolve(raw, data)
                try:
                    if actual is None or expected is None or not OPS[op](actual, expected):
                        return False
                except TypeError:
                    return False
        elif actual != _resolve(cond, data):
            return False
    return True


def render(template: str, data: dict[str, Any]) -> str:
    return string.Formatter().vformat(template, (), _SafeDict(data))


class AlertEngine(Module):
    subscribes = ("event/#",)

    def setup(self) -> None:
        self.rules = self.cfg.get("rules") or []
        for r in self.rules:
            for k in ("event", "kind", "message"):
                if k not in r:
                    raise ValueError(f"alert rule is missing '{k}': {r}")
        self.last_fired: dict[tuple[str, str], float] = {}

    def on_message(self, topic: str, ev: Any) -> None:
        assert isinstance(ev, Event)
        for r in self.rules:
            if r["event"] != ev.type or not matches(r.get("where"), ev.data):
                continue
            key = (r["kind"], render(str(r.get("key", "")), ev.data))
            last = self.last_fired.get(key)
            if last is not None and ev.ts - last < float(r.get("cooldown_s", 300)):
                continue
            self.last_fired[key] = ev.ts
            alert = Alert(kind=r["kind"], message=render(r["message"], ev.data), severity=r.get("severity", "warn"),
                          ts=ev.ts, camera_id=ev.camera_id, data={**ev.data, "event": ev.type}, source=self.name)
            self.publish(f"alert/{r['kind']}", alert)
