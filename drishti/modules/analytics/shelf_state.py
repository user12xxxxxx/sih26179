"""analytics.shelf_state - decides when a slot is really LOW or EMPTY (not just a glitch).

Takes the median of the last few observations per slot and applies hysteresis, so a hand
reaching in or one bad detection does not trigger a restock alert.

Config:
    window: 3             observations to vote over
    low_below: 0.5        occupancy under this -> LOW
    empty_below: 0.15     occupancy under this -> EMPTY
    recover_above: 0.6    must climb back above this to count as restocked

Listens:   event/shelf_observation
Publishes: event/shelf_gap         {slot, row, product, state: LOW|EMPTY, occupancy}
           event/shelf_restocked   {slot, row, product, occupancy}
           event/shelf_status      {slots: {slot_id: state}}   after every observation (for dashboards)
"""

from __future__ import annotations

from collections import deque
from statistics import median
from typing import Any

from drishti.core import Event, Module

ORDER = {"OK": 0, "LOW": 1, "EMPTY": 2}


class ShelfState(Module):
    subscribes = ("event/shelf_observation",)

    def setup(self) -> None:
        self.window = int(self.cfg.get("window", 3))
        self.low = float(self.cfg.get("low_below", 0.5))
        self.empty = float(self.cfg.get("empty_below", 0.15))
        self.recover = float(self.cfg.get("recover_above", 0.6))
        self.hist: dict[str, deque[float]] = {}
        self.state: dict[str, str] = {}

    def classify(self, prev: str, occ: float) -> str:
        if occ < self.empty:
            return "EMPTY"
        if occ < self.low:
            return "LOW"
        if prev != "OK" and occ < self.recover:
            return "LOW"  # partly refilled: not OK until it climbs above recover_above
        return "OK"

    def on_message(self, topic: str, ev: Any) -> None:
        assert isinstance(ev, Event)
        for s in ev.data["slots"]:
            sid = s["id"]
            h = self.hist.setdefault(sid, deque(maxlen=self.window))
            h.append(float(s["occupancy"]))
            if len(h) < self.window:
                continue
            occ = median(h)
            prev = self.state.get(sid, "OK")
            new = self.classify(prev, occ)
            if new == prev:
                continue
            self.state[sid] = new
            base = {"slot": sid, "row": ev.data["row"], "product": s.get("product", ""), "occupancy": round(occ, 3)}
            if ORDER[new] > ORDER[prev]:
                self.emit("shelf_gap", ev.camera_id, ev.ts, state=new, **base)
            elif new == "OK":
                self.emit("shelf_restocked", ev.camera_id, ev.ts, **base)
        self.emit("shelf_status", ev.camera_id, ev.ts, slots=dict(self.state))
