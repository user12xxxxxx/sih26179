"""analytics.line_counter - counts people crossing a line (store entrance), in each direction.

Config:
    camera: entrance
    line: door              name of a line under cameras.<camera>.lines
    ttl_s: 2.0              forget a track after this long unseen

Line definition (drawn with tools/setup_zones.py):
    lines:
      door: {a: [380, 400], b: [900, 400], in_point: [640, 650]}
    in_point is any point inside the store: crossing towards it counts as an entry.

Listens:   track/<camera>
Publishes: event/entry, event/exit   {line, track_id, total_in, total_out}

Reusable for vehicles (ANPR / traffic) unchanged: point it at a road camera.
"""

from __future__ import annotations

from typing import Any

from drishti.core import Module, Tracks
from drishti.core.geometry import segments_intersect, side_of_line


class LineCounter(Module):
    subscribes = ("track/{camera}",)

    def setup(self) -> None:
        self.require("camera", "line")
        ln = self.ctx.line(self.cfg["camera"], self.cfg["line"])
        self.a, self.b = tuple(ln["a"]), tuple(ln["b"])
        self.in_side = side_of_line(tuple(ln["in_point"]), self.a, self.b)
        if self.in_side == 0:
            raise ValueError("in_point must not lie on the line")
        self.ttl = float(self.cfg.get("ttl_s", 2.0))
        self.last: dict[int, tuple[tuple[float, float], int, float]] = {}  # tid -> (point, side, ts)
        self.total_in = 0
        self.total_out = 0

    def on_message(self, topic: str, tracks: Any) -> None:
        assert isinstance(tracks, Tracks)
        for t in tracks.items:
            p = t.anchor
            side = side_of_line(p, self.a, self.b)
            prev = self.last.get(t.track_id)
            if side == 0:  # exactly on the line: keep the previous side
                if prev:
                    self.last[t.track_id] = (prev[0], prev[1], tracks.ts)
                continue
            if prev and prev[1] != side and segments_intersect(prev[0], p, self.a, self.b):
                if side == self.in_side:
                    self.total_in += 1
                    self.emit("entry", tracks.camera_id, tracks.ts, line=self.cfg["line"], track_id=t.track_id,
                              total_in=self.total_in, total_out=self.total_out)
                else:
                    self.total_out += 1
                    self.emit("exit", tracks.camera_id, tracks.ts, line=self.cfg["line"], track_id=t.track_id,
                              total_in=self.total_in, total_out=self.total_out)
            self.last[t.track_id] = (p, side, tracks.ts)
        self.last = {k: v for k, v in self.last.items() if tracks.ts - v[2] <= self.ttl}
