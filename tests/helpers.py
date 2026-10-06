"""Tiny helpers so every module can be tested with hand-made messages - no camera, no model."""

from __future__ import annotations

from typing import Any

from drishti.core import Bus, Context, Track, Tracks


class Recorder:
    """Subscribes to a pattern and keeps everything published on it."""

    def __init__(self, bus: Bus, pattern: str = "#"):
        self.items: list[tuple[str, Any]] = []
        bus.subscribe(pattern, lambda t, m: self.items.append((t, m)))

    def of(self, topic_prefix: str) -> list[Any]:
        return [m for t, m in self.items if t.startswith(topic_prefix)]


def make(cls: type, config: dict | None = None, cameras: dict | None = None, out_dir: Any = "out/test"):
    """Build a module wired to a fresh bus; returns (module, bus, recorder)."""
    from pathlib import Path

    bus = Bus()
    ctx = Context(store={}, cameras=cameras or {}, out_dir=Path(out_dir))
    mod = cls("under_test", bus, config or {}, ctx)
    mod.setup()
    for pattern in mod.topics():
        bus.subscribe(pattern, mod.dispatch)
    return mod, bus, Recorder(bus)


def tracks(cam: str, ts: float, people: dict[int, tuple[float, float]], seq: int = 0) -> Tracks:
    """people: {track_id: (x, y) of the feet}. Boxes are 60 x 150 px above that point."""
    items = [Track(tid, (x - 30, y - 150, x + 30, y)) for tid, (x, y) in people.items()]
    return Tracks(cam, ts, seq, (1280, 720), items)
