"""analytics.queue_monitor - queue length, waiting time and service time per billing counter.

Config:
    camera: checkout
    counters:
      - {id: 1, queue_zone: queue_1, counter_zone: counter_1}
      - {id: 2, queue_zone: queue_2, counter_zone: counter_2}
    interval_s: 15           how often to publish queue_update
    ttl_s: 2.0
    window: 20               average over the last N served customers

Listens:   track/<camera>
Publishes: event/queue_update       {counters: [{id, length, serving, avg_wait_s, avg_service_s}], total_length}
           event/service_complete   {counter, wait_s, service_s}
           event/queue_abandoned    {counter, waited_s}
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import Any

from drishti.core import Module, Tracks
from drishti.core.geometry import point_in_polygon


@dataclass
class _Cust:
    counter: Any = None
    q_enter: float | None = None
    c_enter: float | None = None
    last: float = 0.0
    last_in_counter: float = 0.0
    wait: float | None = None


class QueueMonitor(Module):
    subscribes = ("track/{camera}",)

    def setup(self) -> None:
        self.require("camera", "counters")
        cam = self.cfg["camera"]
        self.counters = [
            (c["id"], self.ctx.zone(cam, c["queue_zone"]), self.ctx.zone(cam, c["counter_zone"])) for c in self.cfg["counters"]
        ]
        self.interval = float(self.cfg.get("interval_s", 15))
        self.ttl = float(self.cfg.get("ttl_s", 2.0))
        n = int(self.cfg.get("window", 20))
        self.waits = {cid: deque(maxlen=n) for cid, _, _ in self.counters}
        self.services = {cid: deque(maxlen=n) for cid, _, _ in self.counters}
        self.cust: dict[int, _Cust] = {}
        self.last_pub = -1e18

    def _finish(self, tid: int, cam: str, ts: float) -> None:
        c = self.cust.pop(tid)
        if c.c_enter is not None:
            service = c.last_in_counter - c.c_enter
            self.services[c.counter].append(service)
            self.emit("service_complete", cam, ts, counter=c.counter, wait_s=round(c.wait or 0.0, 1), service_s=round(service, 1))
        elif c.q_enter is not None and c.last - c.q_enter > 10:
            self.emit("queue_abandoned", cam, ts, counter=c.counter, waited_s=round(c.last - c.q_enter, 1))

    def on_message(self, topic: str, tracks: Any) -> None:
        assert isinstance(tracks, Tracks)
        cam, ts = tracks.camera_id, tracks.ts
        lengths = {cid: 0 for cid, _, _ in self.counters}
        serving = {cid: 0 for cid, _, _ in self.counters}
        seen: set[int] = set()
        for t in tracks.items:
            seen.add(t.track_id)
            c = self.cust.setdefault(t.track_id, _Cust())
            c.last = ts
            for cid, qz, cz in self.counters:
                if point_in_polygon(t.anchor, cz):
                    serving[cid] += 1
                    if c.c_enter is None:
                        c.c_enter, c.counter = ts, cid
                        if c.q_enter is not None:
                            c.wait = ts - c.q_enter
                            self.waits[cid].append(c.wait)
                    c.last_in_counter = ts
                    break
                if point_in_polygon(t.anchor, qz):
                    lengths[cid] += 1
                    if c.q_enter is None:
                        c.q_enter = ts
                    c.counter = cid
                    break
            else:
                if c.c_enter is not None:  # walked away from the counter: service is over
                    self._finish(t.track_id, cam, ts)
        for tid in [k for k, c in self.cust.items() if k not in seen and ts - c.last > self.ttl]:
            self._finish(tid, cam, ts)

        if ts - self.last_pub >= self.interval:
            self.last_pub = ts
            rows = []
            for cid, _, _ in self.counters:
                w, s = self.waits[cid], self.services[cid]
                rows.append({
                    "id": cid, "length": lengths[cid], "serving": serving[cid],
                    "avg_wait_s": round(mean(w), 1) if w else None,
                    "avg_service_s": round(mean(s), 1) if s else None,
                })
            self.emit("queue_update", cam, ts, counters=rows, total_length=sum(lengths.values()))
