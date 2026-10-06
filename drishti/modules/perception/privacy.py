"""perception.redacted_snapshots - occasional debug stills with every person blurred.

The product promise is "no faces, no video stored". This module is the only one allowed to
write images, and it blurs every person box first. Use it while drawing zones and tuning
thresholds; leave it disabled in the demo build.

Config:
    camera: entrance
    every_s: 300
    blur_label: person
    folder: snapshots        (inside out_dir)

Listens: frame/<camera>, det/<camera>
"""

from __future__ import annotations

from typing import Any

import numpy as np

from drishti.core import Detections, Frame, Module


def redact(image: np.ndarray, boxes: list[tuple[float, float, float, float]], pad: int = 10) -> np.ndarray:
    import cv2

    out = image.copy()
    h, w = out.shape[:2]
    for x1, y1, x2, y2 in boxes:
        xa, ya = max(int(x1) - pad, 0), max(int(y1) - pad, 0)
        xb, yb = min(int(x2) + pad, w), min(int(y2) + pad, h)
        if xb > xa and yb > ya:
            roi = out[ya:yb, xa:xb]
            k = max(31, (min(roi.shape[:2]) // 2) | 1)
            out[ya:yb, xa:xb] = cv2.GaussianBlur(roi, (k, k), 0)
    return out


class RedactedSnapshots(Module):
    subscribes = ("frame/{camera}", "det/{camera}")

    def setup(self) -> None:
        self.require("camera")
        self.every = float(self.cfg.get("every_s", 300))
        self.label = self.cfg.get("blur_label", "person")
        self.folder = self.ctx.out_dir / self.cfg.get("folder", "snapshots")
        self.folder.mkdir(parents=True, exist_ok=True)
        self.last_saved = -1e18
        self.pending: Frame | None = None

    def on_message(self, topic: str, msg: Any) -> None:
        import cv2

        if isinstance(msg, Frame):
            if msg.ts - self.last_saved >= self.every:
                self.pending = msg  # wait for this frame's detections before saving
        elif isinstance(msg, Detections) and self.pending is not None and msg.seq == self.pending.seq:
            boxes = [d.bbox for d in msg.items if d.label == self.label]
            path = self.folder / f"{msg.camera_id}_{int(msg.ts)}.jpg"
            cv2.imwrite(str(path), redact(self.pending.image, boxes))
            self.last_saved, self.pending = msg.ts, None
