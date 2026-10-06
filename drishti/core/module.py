"""Base classes every module inherits from.

A module is one class that
  * declares which topics it listens to (``subscribes``),
  * reacts in ``on_message``,
  * publishes results with ``publish`` / ``emit`` / ``raise_alert``,
  * reads its settings from ``self.cfg`` (the ``config:`` block in the pipeline YAML).

It never imports another module. Swapping an implementation means writing a new module
that uses the same topics and changing ``use:`` in the YAML.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bus import Bus
from .types import Alert, Event, Metric, Point


@dataclass
class Context:
    """Shared, read-only store description: camera geometry drawn with tools/setup_zones.py."""

    store: dict[str, Any] = field(default_factory=dict)
    cameras: dict[str, dict[str, Any]] = field(default_factory=dict)
    out_dir: Path = Path("out")

    def camera(self, camera_id: str) -> dict[str, Any]:
        if camera_id not in self.cameras:
            raise KeyError(f"camera '{camera_id}' is not defined under 'cameras:' in the config")
        return self.cameras[camera_id]

    def zone(self, camera_id: str, name: str) -> list[Point]:
        zones = self.camera(camera_id).get("zones", {})
        if name not in zones:
            raise KeyError(f"zone '{name}' is not defined for camera '{camera_id}'")
        return [tuple(p) for p in zones[name]]

    def line(self, camera_id: str, name: str) -> dict[str, Any]:
        lines = self.camera(camera_id).get("lines", {})
        if name not in lines:
            raise KeyError(f"line '{name}' is not defined for camera '{camera_id}'")
        return lines[name]

    def shelves(self, camera_id: str) -> list[dict[str, Any]]:
        return self.camera(camera_id).get("shelves", [])


class Module:
    """Base class for everything that reacts to messages."""

    #: Topic patterns to subscribe to. ``{camera}`` etc. are filled from the config block.
    subscribes: tuple[str, ...] = ()

    def __init__(self, name: str, bus: Bus, config: dict[str, Any] | None = None, ctx: Context | None = None):
        self.name = name
        self.bus = bus
        self.cfg: dict[str, Any] = dict(config or {})
        self.ctx = ctx or Context()
        self.log = logging.getLogger(f"drishti.{name}")
        self._lock = threading.Lock()
        self.errors = 0

    # -- lifecycle -------------------------------------------------------------------------
    def setup(self) -> None:
        """Load models, open files, validate config. Import optional libraries here."""

    def on_message(self, topic: str, msg: Any) -> None:
        """Handle one message. Keep it fast; heavy work belongs in a perception module."""

    def teardown(self) -> None:
        """Flush and close resources."""

    # -- wiring ----------------------------------------------------------------------------
    def topics(self) -> list[str]:
        patterns = self.cfg.get("subscribe") or self.subscribes
        return [p.format(**self.cfg) for p in patterns]

    def dispatch(self, topic: str, msg: Any) -> None:
        # One broken module must not take the whole store down: log and carry on.
        with self._lock:
            try:
                self.on_message(topic, msg)
            except Exception:  # noqa: BLE001
                self.errors += 1
                self.log.exception("error while handling %s", topic)

    # -- helpers ---------------------------------------------------------------------------
    def publish(self, topic: str, msg: Any) -> None:
        self.bus.publish(topic, msg)

    def emit(self, type_: str, camera_id: str, ts: float, **data: Any) -> Event:
        ev = Event(type=type_, camera_id=camera_id, ts=ts, data=data, source=self.name)
        self.publish(f"event/{type_}", ev)
        return ev

    def raise_alert(self, kind: str, message: str, severity: str, ts: float, camera_id: str = "", **data: Any) -> Alert:
        al = Alert(kind=kind, message=message, severity=severity, ts=ts, camera_id=camera_id, data=data, source=self.name)
        self.publish(f"alert/{kind}", al)
        return al

    def metric(self, name: str, value: float, ts: float, camera_id: str = "") -> None:
        self.publish(f"metric/{name}", Metric(name=name, value=float(value), ts=ts, camera_id=camera_id, source=self.name))

    def require(self, *keys: str) -> None:
        missing = [k for k in keys if k not in self.cfg]
        if missing:
            raise ValueError(f"{self.name}: missing config key(s): {', '.join(missing)}")


class Source(Module):
    """A module that produces data on its own thread: camera, video file, simulator."""

    def run(self, stop: threading.Event) -> None:
        raise NotImplementedError
