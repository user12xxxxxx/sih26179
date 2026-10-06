"""source.synthetic - a fake store: shoppers enter, browse, queue, pay and leave.

No camera and no model needed. It publishes the same ``track/<camera>`` messages a real
detector + tracker would, so analytics, alerts, the event store, reports and the dashboard
can all be built and tested before the vision side works. Time is simulated: an hour of
store time runs in about a second.

It also closes the loop for demos: when an ``open_counter`` alert arrives, the simulated
manager opens the recommended counters after ``response_s`` seconds.

Publishes: track/<entrance_camera>, track/<checkout_camera>, event/counter_status
Listens:   alert/open_counter
"""

from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from drishti.core import Source, Track, Tracks

DEFAULTS: dict[str, Any] = {
    "entrance_camera": "entrance",
    "checkout_camera": "checkout",
    "duration_min": 60,
    "fps": 2,  # track updates per simulated second
    "start": "2026-10-01T17:00:00+05:30",
    "arrivals_per_min": 1.0,
    "rush": {"start_min": 20, "end_min": 32, "arrivals_per_min": 3.0},
    "shop_time_min": [6, 12],
    "checkout_prob": 0.85,
    "promo_stop_prob": 0.4,
    "promo_stop_s": [5, 40],
    "service_s": 40,
    "counters_open": 1,
    "max_counters": 3,
    "respond_to_alerts": True,
    "response_s": 60,
    "realtime": False,
    "speed": 1.0,
    "seed": 7,
}

FRAME = (1280, 720)
DOOR_Y = 400  # matches the door line in configs/synthetic_demo.yaml
PROMO_POINT = (1100.0, 600.0)
COUNTER_X = [250.0, 550.0, 850.0, 1150.0]
COUNTER_Y = 640.0
QUEUE_FRONT_Y = 520.0
QUEUE_STEP = 45.0


def _box(x: float, y: float) -> tuple[float, float, float, float]:
    return (x - 30.0, y - 150.0, x + 30.0, y)


def _lerp(a: tuple[float, float], b: tuple[float, float], f: float) -> tuple[float, float]:
    f = min(max(f, 0.0), 1.0)
    return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)


@dataclass
class Shopper:
    sid: int
    t_enter: float
    shop_s: float
    to_checkout: bool
    promo_s: float
    lane_x: float
    state: str = "entering"  # entering -> shopping -> queued -> serving -> walking_out -> exiting -> done
    t_state: float = 0.0
    counter: int = -1
    service_s: float = 0.0
    ent_track: int = -1
    chk_track: int = -1


class SyntheticStore(Source):
    subscribes = ("alert/open_counter",)

    def setup(self) -> None:
        c = {**DEFAULTS, **self.cfg}
        c["rush"] = {**DEFAULTS["rush"], **(self.cfg.get("rush") or {})}
        self.c = c
        self.rng = random.Random(c["seed"])
        self.dt = 1.0 / float(c["fps"])
        self.t0 = datetime.fromisoformat(c["start"]).timestamp()
        self.open = int(c["counters_open"])
        self.pending_open: list[tuple[float, int]] = []
        self._pending_lock = threading.Lock()
        self.now = self.t0

    # -- alert loop-back ------------------------------------------------------------------
    def on_message(self, topic: str, msg: Any) -> None:
        if not self.c["respond_to_alerts"]:
            return
        want = int(msg.data.get("recommended_counters", self.open + 1))
        with self._pending_lock:
            self.pending_open.append((self.now + float(self.c["response_s"]), min(want, int(self.c["max_counters"]))))

    # -- helpers ------------------------------------------------------------------------------
    def _rate_per_s(self, t: float) -> float:
        m = (t - self.t0) / 60.0
        r = self.c["rush"]
        per_min = r["arrivals_per_min"] if r["start_min"] <= m < r["end_min"] else self.c["arrivals_per_min"]
        return per_min / 60.0

    def _poisson(self, lam: float) -> int:
        l_exp, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= self.rng.random()
            if p <= l_exp:
                return k
            k += 1

    def _status(self, t: float) -> None:
        self.emit("counter_status", self.c["checkout_camera"], t, open_counters=self.open)

    def _open_counters(self, n: int, queues: list[list[Shopper]], t: float) -> None:
        if n <= self.open:
            return
        self.open = n
        # people at the back of long queues walk over to the newly opened counters
        while True:
            lengths = [len(q) for q in queues[: self.open]]
            longest, shortest = lengths.index(max(lengths)), lengths.index(min(lengths))
            if lengths[longest] - lengths[shortest] <= 1:
                break
            s = queues[longest].pop()
            s.counter = shortest
            queues[shortest].append(s)
        self._status(t)

    # -- main loop ----------------------------------------------------------------------------
    def run(self, stop: threading.Event) -> None:
        c = self.c
        ent_cam, chk_cam = c["entrance_camera"], c["checkout_camera"]
        n_ticks = int(float(c["duration_min"]) * 60.0 / self.dt)
        shoppers: list[Shopper] = []
        queues: list[list[Shopper]] = [[] for _ in range(int(c["max_counters"]))]
        serving: list[Shopper | None] = [None] * int(c["max_counters"])
        next_id = 1
        next_track = {ent_cam: 1, chk_cam: 1}
        wall0 = time.monotonic()
        self._status(self.t0)

        def new_track(cam: str) -> int:
            tid = next_track[cam]
            next_track[cam] += 1
            return tid

        for tick in range(n_ticks):
            if stop.is_set():
                break
            t = self.t0 + tick * self.dt
            self.now = t

            with self._pending_lock:
                due = [p for p in self.pending_open if p[0] <= t]
                self.pending_open = [p for p in self.pending_open if p[0] > t]
            for _, n in due:
                self._open_counters(n, queues, t)

            # arrivals
            for _ in range(self._poisson(self._rate_per_s(t) * self.dt)):
                lo, hi = c["shop_time_min"]
                promo = self.rng.uniform(*c["promo_stop_s"]) if self.rng.random() < c["promo_stop_prob"] else 0.0
                s = Shopper(
                    sid=next_id, t_enter=t, shop_s=self.rng.uniform(lo, hi) * 60.0,
                    to_checkout=self.rng.random() < c["checkout_prob"], promo_s=promo,
                    lane_x=self.rng.uniform(460.0, 820.0), t_state=t, ent_track=new_track(ent_cam),
                )
                next_id += 1
                shoppers.append(s)

            # state machine
            for s in shoppers:
                el = t - s.t_state
                if s.state == "entering":
                    total = 4.0 if s.promo_s == 0 else 3.0 + 2.0 + s.promo_s + 1.5
                    if el >= total:
                        s.state, s.t_state = "shopping", t
                elif s.state == "shopping" and t - s.t_enter >= s.shop_s:
                    if s.to_checkout:
                        k = min(range(self.open), key=lambda i: len(queues[i]) + (serving[i] is not None))
                        s.counter, s.state, s.t_state = k, "queued", t
                        s.chk_track = new_track(chk_cam)
                        queues[k].append(s)
                    else:
                        s.state, s.t_state = "walking_out", t
                elif s.state == "serving" and el >= s.service_s:
                    serving[s.counter] = None
                    s.state, s.t_state = "walking_out", t
                elif s.state == "walking_out" and el >= 10.0:
                    s.state, s.t_state = "exiting", t
                    s.ent_track = new_track(ent_cam)
                elif s.state == "exiting" and el >= 4.0:
                    s.state = "done"

            for k in range(self.open):
                if serving[k] is None and queues[k]:
                    s = queues[k].pop(0)
                    s.state, s.t_state = "serving", t
                    s.service_s = min(max(self.rng.expovariate(1.0 / c["service_s"]), 10.0), 240.0)
                    serving[k] = s
            shoppers = [s for s in shoppers if s.state != "done"]

            # what the two cameras would see
            ent_items: list[Track] = []
            chk_items: list[Track] = []
            for s in shoppers:
                el = t - s.t_state
                if s.state == "entering":
                    if s.promo_s == 0:
                        p = _lerp((s.lane_x, 60.0), (s.lane_x, 700.0), el / 4.0)
                    elif el < 3.0:
                        p = _lerp((s.lane_x, 60.0), (s.lane_x, 500.0), el / 3.0)
                    elif el < 5.0:
                        p = _lerp((s.lane_x, 500.0), PROMO_POINT, (el - 3.0) / 2.0)
                    elif el < 5.0 + s.promo_s:
                        p = PROMO_POINT
                    else:
                        p = _lerp(PROMO_POINT, (1300.0, 650.0), (el - 5.0 - s.promo_s) / 1.5)
                    if p[0] < FRAME[0]:
                        ent_items.append(Track(s.ent_track, _box(*p)))
                elif s.state == "exiting":
                    p = _lerp((s.lane_x, 700.0), (s.lane_x, 60.0), el / 4.0)
                    ent_items.append(Track(s.ent_track, _box(*p)))
                elif s.state == "serving":
                    chk_items.append(Track(s.chk_track, _box(COUNTER_X[s.counter], COUNTER_Y)))
            for k, q in enumerate(queues):
                for i, s in enumerate(q):
                    y = max(QUEUE_FRONT_Y - i * QUEUE_STEP, 160.0)
                    chk_items.append(Track(s.chk_track, _box(COUNTER_X[k], y)))

            self.publish(f"track/{ent_cam}", Tracks(ent_cam, t, tick, FRAME, ent_items, {"synthetic": True}))
            self.publish(f"track/{chk_cam}", Tracks(chk_cam, t, tick, FRAME, chk_items, {"synthetic": True}))

            if c["realtime"]:
                target = wall0 + (tick + 1) * self.dt / float(c["speed"])
                delay = target - time.monotonic()
                if delay > 0:
                    stop.wait(delay)
        self.log.info("simulation finished: %d shoppers entered, %d counters open at the end", next_id - 1, self.open)
