"""analytics.heatmap - where people spend time, as a grid of person-seconds.

Config:
    camera: entrance
    grid: [64, 36]            columns x rows
    interval_s: 300           publish a snapshot every 5 minutes of store time
    save_png: false           also write out/heatmap_<camera>.png (needs OpenCV; off by default -
                              out/ must stay free of images, see the README privacy rule)
    floor_plan:               optional: map camera pixels onto a floor plan
      src: [[x,y] x4]         four points in the camera image
      dst: [[u,v] x4]         the same four points on the plan
      size: [800, 600]        plan size in the same units as dst

Listens:   track/<camera>
Publishes: event/heatmap   {grid (list of rows, values 0-1), cols, rows, person_seconds}

Reusable for traffic density maps (vehicles instead of people).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from drishti.core import Module, Tracks
from drishti.core.geometry import Homography


class Heatmap(Module):
    subscribes = ("track/{camera}",)

    def setup(self) -> None:
        self.require("camera")
        self.cols, self.rows = (int(v) for v in self.cfg.get("grid", [64, 36]))
        self.grid = np.zeros((self.rows, self.cols), dtype=np.float64)
        self.interval = float(self.cfg.get("interval_s", 300))
        fp = self.cfg.get("floor_plan")
        self.h = Homography(fp["src"], fp["dst"]) if fp else None
        self.space = tuple(fp["size"]) if fp else None
        self.last_ts: float | None = None
        self.last_pub: float | None = None
        self.prev: dict[int, tuple[float, float, float]] = {}  # track id -> (x, y, ts)

    def _add(self, x: float, y: float, w: float, h: float, amount: float) -> None:
        c, r = int(x / w * self.cols), int(y / h * self.rows)
        if 0 <= c < self.cols and 0 <= r < self.rows:
            self.grid[r, c] += amount

    def on_message(self, topic: str, tracks: Any) -> None:
        assert isinstance(tracks, Tracks)
        ts = tracks.ts
        dt = 0.0 if self.last_ts is None else min(ts - self.last_ts, 2.0)  # cap gaps (dropped frames)
        self.last_ts = ts
        if self.last_pub is None:
            self.last_pub = ts
        w, h = self.space or tracks.frame_size
        cell = min(w / self.cols, h / self.rows)
        for t in tracks.items:
            x, y = self.h(t.anchor) if self.h else t.anchor
            prev = self.prev.get(t.track_id)
            if prev and ts - prev[2] <= 1.5:
                # spread the time along the path walked since the last update, so low frame
                # rates do not leave gaps between the cells a person actually crossed
                steps = max(1, int(((x - prev[0]) ** 2 + (y - prev[1]) ** 2) ** 0.5 / cell))
                for k in range(1, steps + 1):
                    f = k / steps
                    self._add(prev[0] + (x - prev[0]) * f, prev[1] + (y - prev[1]) * f, w, h, dt / steps)
            else:
                self._add(x, y, w, h, dt)
            self.prev[t.track_id] = (x, y, ts)
        self.prev = {k: v for k, v in self.prev.items() if ts - v[2] <= 1.5}
        if ts - self.last_pub >= self.interval:
            self.last_pub = ts
            self.flush(tracks.camera_id, ts)

    def flush(self, cam: str, ts: float) -> None:
        total = float(self.grid.sum())
        norm = self.grid / self.grid.max() if self.grid.max() > 0 else self.grid
        self.emit("heatmap", cam, ts, cols=self.cols, rows=self.rows, person_seconds=round(total, 1),
                  grid=np.round(norm, 3).tolist())
        if self.cfg.get("save_png", False):
            try:
                import cv2

                shown = np.log1p(40 * norm) / np.log1p(40)  # log scale so quiet areas stay visible
                big = cv2.resize(shown.astype(np.float32), (self.cols * 10, self.rows * 10), interpolation=cv2.INTER_CUBIC)
                img = cv2.applyColorMap((np.clip(big, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(str(self.ctx.out_dir / f"heatmap_{cam}.png"), img)
            except ImportError:
                pass

    def teardown(self) -> None:
        if self.last_ts is not None and self.grid.sum() > 0:
            self.flush(self.cfg["camera"], self.last_ts)
