"""analytics.queue_forecast - predicts the billing rush before it reaches the counters.

Idea: people who walked in during the last few minutes will reach checkout soon. Each entry
contributes the probability that it arrives at the counters within the forecast horizon,
given typical shopping time (a uniform range, tunable from real data). That gives the
expected arrival rate lambda; recent service times give the service rate mu. The Erlang C
queueing model then says how long a new customer would wait with c counters open, and the
module recommends the smallest c that keeps the wait under the target.

No identities are linked across cameras - only counts - so this stays privacy-safe.

Config:
    camera: checkout              camera id to stamp on the forecast events
    entry_event: entry            event type counted as an arrival (from analytics.line_counter)
    shop_time_min: [6, 12]        typical time between entering and reaching checkout
    checkout_prob: 0.85           share of visitors who buy something
    horizon_min: 5
    target_wait_s: 180
    max_counters: 3
    open_counters: 1              initial value; event/counter_status updates it
    default_service_s: 60         typical billing time; blended with measured times
    prior_weight: 10              how many "virtual customers" the default counts as
    interval_s: 30

Listens:   event/<entry_event>, event/service_complete, event/queue_update, event/counter_status
Publishes: event/queue_forecast {expected_arrivals, lambda_per_min, mu_per_min, queue_length,
            open_counters, predicted_wait_s, predicted_wait_min, recommended_counters, recommended_wait_s}
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Iterable

from drishti.core import Event, Module


def erlang_c(c: int, load: float) -> float:
    """Probability that an arriving customer has to wait (M/M/c). load = lambda / mu."""
    if c <= load:
        return 1.0
    term = 1.0
    acc = 1.0  # sum_{k=0}^{c-1} load^k / k!
    for k in range(1, c):
        term *= load / k
        acc += term
    top = term * load / c * c / (c - load)  # load^c / c! * c / (c - load)
    return top / (acc + top)


def expected_wait_s(lam: float, mu: float, c: int, queue_len: int = 0) -> float:
    """Expected wait (seconds) for a new customer: steady-state Erlang C wait + clearing the backlog.

    lam and mu are per second. Returns math.inf when the counters cannot keep up.
    """
    if c <= 0 or mu <= 0:
        return math.inf
    backlog = queue_len / (c * mu)
    if lam <= 0:
        return backlog
    load = lam / mu
    if c <= load:
        return math.inf
    return erlang_c(c, load) / (c * mu - lam) + backlog


def expected_arrivals(entries: Iterable[float], now: float, lo_s: float, hi_s: float, horizon_s: float) -> float:
    """Expected number of recent entrants who reach checkout within [now, now + horizon].

    Shopping time L ~ Uniform(lo, hi). For someone who entered s seconds ago the chance of
    arriving in the window is P(s < L <= s + horizon). We cannot see who already reached
    the counters (no identities across cameras), so the unconditional probability is used.
    """
    n = 0.0
    width = hi_s - lo_s
    for e in entries:
        s = now - e
        if s >= hi_s or s < 0:
            continue
        if width <= 0:
            n += 1.0 if s < lo_s <= s + horizon_s else 0.0
            continue
        n += max(0.0, min(s + horizon_s, hi_s) - max(s, lo_s)) / width
    return n


class QueueForecast(Module):
    subscribes = ("event/{entry_event}", "event/service_complete", "event/queue_update", "event/counter_status")

    def setup(self) -> None:
        self.cfg.setdefault("entry_event", "entry")
        lo, hi = self.cfg.get("shop_time_min", [6, 12])
        self.lo_s, self.hi_s = float(lo) * 60, float(hi) * 60
        self.p = float(self.cfg.get("checkout_prob", 0.85))
        self.h_s = float(self.cfg.get("horizon_min", 5)) * 60
        self.target = float(self.cfg.get("target_wait_s", 180))
        self.max_c = int(self.cfg.get("max_counters", 3))
        self.open = int(self.cfg.get("open_counters", 1))
        self.interval = float(self.cfg.get("interval_s", 30))
        self.default_service = float(self.cfg.get("default_service_s", 60))
        self.prior_weight = float(self.cfg.get("prior_weight", 10))
        self.cam = self.cfg.get("camera", "checkout")
        self.entries: deque[float] = deque()
        self.services: deque[float] = deque(maxlen=30)
        self.queue_len = 0
        self.last_run = -1e18

    def on_message(self, topic: str, ev: Any) -> None:
        assert isinstance(ev, Event)
        if ev.type == self.cfg["entry_event"]:
            self.entries.append(ev.ts)
        elif ev.type == "service_complete":
            self.services.append(float(ev.data["service_s"]))
        elif ev.type == "queue_update":
            self.queue_len = int(ev.data.get("total_length", 0))
        elif ev.type == "counter_status":
            self.open = int(ev.data["open_counters"])
        while self.entries and ev.ts - self.entries[0] > self.hi_s:
            self.entries.popleft()
        if ev.ts - self.last_run >= self.interval:
            self.last_run = ev.ts
            self.forecast(ev.ts)

    def forecast(self, now: float) -> dict[str, Any]:
        n = self.p * expected_arrivals(self.entries, now, self.lo_s, self.hi_s, self.h_s)
        lam = n / self.h_s
        # Blend measured billing times with the configured typical value, so the first few
        # (possibly lucky, fast) customers do not make the counters look faster than they are.
        k = self.prior_weight
        service = (sum(self.services) + k * self.default_service) / (len(self.services) + k)
        mu = 1.0 / service
        wait_now = expected_wait_s(lam, mu, self.open, self.queue_len)
        rec, rec_wait = self.max_c, expected_wait_s(lam, mu, self.max_c, self.queue_len)
        for c in range(1, self.max_c + 1):
            w = expected_wait_s(lam, mu, c, self.queue_len)
            if w <= self.target:
                rec, rec_wait = c, w
                break

        def cap(x: float) -> float:
            return round(min(x, 3600.0), 1)

        data = {
            "horizon_min": round(self.h_s / 60, 1), "expected_arrivals": round(n, 2),
            "lambda_per_min": round(lam * 60, 3), "mu_per_min": round(mu * 60, 3),
            "queue_length": self.queue_len, "open_counters": self.open,
            "predicted_wait_s": cap(wait_now), "predicted_wait_min": round(min(wait_now, 3600.0) / 60, 1),
            "recommended_counters": max(rec, 1), "recommended_wait_s": cap(rec_wait),
        }
        self.emit("queue_forecast", self.cam, now, **data)
        return data
