"""perception.shelf_reference - shelf fallback with zero training.

Instead of running a product detector, this compares the live frame to a "full" and an "empty"
reference photo taken once at setup time. It publishes the exact same event/shelf_observation
contract analytics.shelf_gaps does, so analytics.shelf_state and restock alerts work unchanged -
swap which one is listed in the pipeline YAML and nothing downstream changes.

Config:
    camera: shelf_a
    full_ref: snapshots/shelf_a_full.jpg     photo taken with every slot fully stocked
    empty_ref: snapshots/shelf_a_empty.jpg   photo taken with every slot emptied out
    interval_s: 30                           shelves change slowly; no need to check every frame
    gap_below: 0.15                          a slot below this occupancy is reported as a gap

Reads the shelf layout (rows, slots) from cameras.<camera>.shelves - the same config
analytics.shelf_gaps uses (drawn with tools/setup_zones.py). Both reference photos must be the
same size as the camera's frames, and taken from the same fixed mount.

Listens:   frame/<camera>
Publishes: event/shelf_observation   {row, slots: [{id, product, occupancy}], gaps: [[x1, x2], ...]}

Limitation: with no product boxes to reason about, it can't tell a shopper's hand from an empty
slot - a hand reaching into a slot briefly reads as low occupancy. analytics.shelf_state's
window/hysteresis absorbs single bad observations like that, so this is fine as the demo fallback.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from drishti.core import Frame, Module


def slot_occupancy(now: np.ndarray, full: np.ndarray, empty: np.ndarray) -> float:
    """now/full/empty: same-shape grayscale crops of one slot. 1.0 reads as "full", 0.0 as "empty"."""
    d_full = float(np.abs(now.astype(np.int16) - full.astype(np.int16)).mean())
    d_empty = float(np.abs(now.astype(np.int16) - empty.astype(np.int16)).mean())
    total = d_full + d_empty
    if total < 1e-6:
        return 1.0  # references are identical: can't tell full from empty, assume stocked
    return max(0.0, min(1.0, d_empty / total))


class ShelfReference(Module):
    subscribes = ("frame/{camera}",)

    def setup(self) -> None:
        self.require("camera", "full_ref", "empty_ref")
        import cv2

        self.rows = self.ctx.shelves(self.cfg["camera"])
        if not self.rows:
            raise ValueError(f"no shelves defined for camera {self.cfg['camera']}")
        full = cv2.imread(str(self.cfg["full_ref"]), cv2.IMREAD_GRAYSCALE)
        empty = cv2.imread(str(self.cfg["empty_ref"]), cv2.IMREAD_GRAYSCALE)
        if full is None:
            raise FileNotFoundError(f"cannot open full_ref '{self.cfg['full_ref']}'")
        if empty is None:
            raise FileNotFoundError(f"cannot open empty_ref '{self.cfg['empty_ref']}'")
        if full.shape != empty.shape:
            raise ValueError("full_ref and empty_ref must be the same size - retake them together")
        self.full, self.empty = full, empty
        self.interval = float(self.cfg.get("interval_s", 30))
        self.gap_below = float(self.cfg.get("gap_below", 0.15))
        self.last_ts = -1e18

    def on_message(self, topic: str, frame: Any) -> None:
        assert isinstance(frame, Frame)
        if frame.ts - self.last_ts < self.interval:
            return
        self.last_ts = frame.ts
        import cv2

        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        if gray.shape != self.full.shape:
            gray = cv2.resize(gray, (self.full.shape[1], self.full.shape[0]))
        for row in self.rows:
            y1, y2 = int(row["rect"][1]), int(row["rect"][3])
            slots, gaps = [], []
            for s in row.get("slots", []):
                lo, hi = int(s["x"][0]), int(s["x"][1])
                occ = slot_occupancy(gray[y1:y2, lo:hi], self.full[y1:y2, lo:hi], self.empty[y1:y2, lo:hi])
                slots.append({"id": s["id"], "product": s.get("product", ""), "occupancy": round(occ, 3)})
                if occ < self.gap_below:
                    gaps.append([s["x"][0], s["x"][1]])
            self.emit("shelf_observation", frame.camera_id, frame.ts, row=row["id"], slots=slots, gaps=gaps)
