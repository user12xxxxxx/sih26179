"""source.rtsp - a live IP camera or a phone running an IP-camera app.

Config:
    camera: checkout
    url: rtsp://user:pass@192.168.1.50:554/stream1     (Tapo C200: stream1 = 1080p, stream2 = 360p)
    max_fps: 10          drop frames above this rate to save CPU/NPU
    reconnect_s: 3

Publishes: frame/<camera>   (local only)

Later, on the Qualcomm board, replace this with a GStreamer/IM SDK source that decodes on the
chip's video engine; nothing downstream has to change.
"""

from __future__ import annotations

import threading
import time

from drishti.core import Frame, Source


class RtspSource(Source):
    def setup(self) -> None:
        self.require("camera", "url")
        import cv2  # noqa: F401

    def run(self, stop: threading.Event) -> None:
        import cv2

        cam = self.cfg["camera"]
        min_gap = 1.0 / float(self.cfg.get("max_fps", 10))
        seq, last = 0, 0.0
        while not stop.is_set():
            cap = cv2.VideoCapture(self.cfg["url"], cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self.log.warning("cannot open %s, retrying", cam)
                stop.wait(float(self.cfg.get("reconnect_s", 3)))
                continue
            while not stop.is_set():
                ok, img = cap.read()
                if not ok:
                    self.log.warning("stream %s dropped, reconnecting", cam)
                    break
                now = time.time()
                if now - last < min_gap:
                    continue
                last = now
                self.publish(f"frame/{cam}", Frame(cam, now, seq, img))
                seq += 1
            cap.release()
            stop.wait(float(self.cfg.get("reconnect_s", 3)))
