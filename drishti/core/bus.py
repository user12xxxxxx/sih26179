"""In-process publish/subscribe bus with MQTT-style topics.

The same topic names work on a real MQTT broker (see ``output.mqtt_bridge``), so a module
can later move to another process or another device without changing its code.

Wildcards follow MQTT: ``+`` matches one level, ``#`` matches everything below.
    "track/+"   matches "track/entrance" but not "track/entrance/x"
    "event/#"   matches "event/entry", "event/queue/update", ...
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from typing import Any, Callable

Handler = Callable[[str, Any], None]

# Topics that must never leave the process: raw images. The MQTT bridge checks this list.
LOCAL_ONLY_PREFIXES: tuple[str, ...] = ("frame/",)

log = logging.getLogger("drishti.bus")


def topic_matches(pattern: str, topic: str) -> bool:
    p_parts = pattern.split("/")
    t_parts = topic.split("/")
    for i, part in enumerate(p_parts):
        if part == "#":
            return True
        if i >= len(t_parts):
            return False
        if part != "+" and part != t_parts[i]:
            return False
    return len(p_parts) == len(t_parts)


def is_local_only(topic: str) -> bool:
    return topic.startswith(LOCAL_ONLY_PREFIXES)


class Bus:
    """Synchronous bus: ``publish`` calls every matching handler before returning.

    Synchronous delivery keeps replays deterministic (same input clip -> same events),
    which is what makes the unit tests and the benchmark trustworthy. A slow module
    (e.g. a detector) should keep its own work queue if it must not block its source.
    """

    def __init__(self) -> None:
        self._subs: list[tuple[str, Handler]] = []
        self._lock = threading.RLock()
        self.counts: Counter[str] = Counter()  # messages per top-level topic, for health checks

    def subscribe(self, pattern: str, handler: Handler) -> None:
        with self._lock:
            self._subs.append((pattern, handler))

    def unsubscribe(self, handler: Handler) -> None:
        # `==`, not `is`: `obj.method` builds a fresh bound-method object on every attribute
        # access, so the object Pipeline.stop() passes here is never identical to the one
        # Pipeline.start() subscribed. Bound methods still compare equal (same func, same self).
        with self._lock:
            self._subs = [(p, h) for p, h in self._subs if h != handler]

    def publish(self, topic: str, msg: Any) -> None:
        self.counts[topic.split("/", 1)[0]] += 1
        with self._lock:
            subs = list(self._subs)
        for pattern, handler in subs:
            if topic_matches(pattern, topic):
                handler(topic, msg)
