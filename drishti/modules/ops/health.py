"""ops.health - frames per second per camera, model latency, and (on Linux boards) SoC temperature.

These numbers go on the "feasibility" slide and catch problems early (a camera silently
dropping to 2 fps ruins counting accuracy).

Config:
    every_s: 10                 wall-clock seconds between reports
    thermal_zone: /sys/class/thermal/thermal_zone0/temp    optional

Listens:   frame/#, det/#
Publishes: metric/fps, metric/infer_ms, metric/soc_temp_c
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from drishti.core import Detections, Frame, Module


class Health(Module):
    subscribes = ("frame/#", "det/#")

    def setup(self) -> None:
        self.every = float(self.cfg.get("every_s", 10))
        self.frames: dict[str, int] = defaultdict(int)
        self.infer: dict[str, list[float]] = defaultdict(list)
        self.window_start: float | None = None
        self.thermal = Path(self.cfg.get("thermal_zone", "/sys/class/thermal/thermal_zone0/temp"))

    def on_message(self, topic: str, msg: Any) -> None:
        if isinstance(msg, Frame):
            self.frames[msg.camera_id] += 1
        elif isinstance(msg, Detections) and "infer_ms" in msg.meta:
            self.infer[msg.camera_id].append(float(msg.meta["infer_ms"]))
        ts = msg.ts
        now = time.monotonic()  # fps = processing speed, so measure it on the wall clock
        if self.window_start is None:
            self.window_start = now
        span = now - self.window_start
        if span >= self.every:
            for cam, n in self.frames.items():
                self.metric("fps", n / span, ts, cam)
            for cam, xs in self.infer.items():
                if xs:
                    self.metric("infer_ms", sum(xs) / len(xs), ts, cam)
            if self.thermal.exists():
                try:
                    self.metric("soc_temp_c", int(self.thermal.read_text().strip()) / 1000.0, ts)
                except (OSError, ValueError):
                    pass
            self.frames.clear()
            self.infer.clear()
            self.window_start = now
