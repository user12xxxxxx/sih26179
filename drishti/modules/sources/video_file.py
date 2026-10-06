"""source.video_file - replay a recorded clip as if it were a live camera.

Replays are how you test: record the mock store once, then every change to any module can be
re-run on exactly the same footage. Timestamps come from the clip (start + frame / fps), so
results are identical run after run.

Config:
    camera: entrance            camera id used in topics
    path: clips/entrance.mp4
    start: "2026-10-01T17:00:00+05:30"   optional wall time of the first frame
    realtime: false             true = pace at the clip's fps (for live-looking demos)
    stride: 1                   use every n-th frame (2 = half the frame rate)
    loop: false

Publishes: frame/<camera>   (local only)
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

from drishti.core import Frame, Source


class VideoFileSource(Source):
    def setup(self) -> None:
        self.require("camera", "path")
        import cv2  # noqa: F401  (fail early if OpenCV is missing)

    def run(self, stop: threading.Event) -> None:
        import cv2

        cam = self.cfg["camera"]
        stride = int(self.cfg.get("stride", 1))
        start = self.cfg.get("start")
        t0 = datetime.fromisoformat(start).timestamp() if start else time.time()
        seq = 0
        while not stop.is_set():
            cap = cv2.VideoCapture(str(self.cfg["path"]))
            if not cap.isOpened():
                raise FileNotFoundError(f"cannot open {self.cfg['path']}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            idx = 0
            wall0 = time.monotonic()
            while not stop.is_set():
                ok, img = cap.read()
                if not ok:
                    break
                if idx % stride == 0:
                    ts = t0 + idx / fps
                    self.publish(f"frame/{cam}", Frame(cam, ts, seq, img))
                    seq += 1
                    if self.cfg.get("realtime", False):
                        delay = wall0 + idx / fps - time.monotonic()
                        if delay > 0:
                            stop.wait(delay)
                idx += 1
            cap.release()
            if not self.cfg.get("loop", False):
                break
            t0 += idx / fps
        self.log.info("clip finished after %d frames", seq)
