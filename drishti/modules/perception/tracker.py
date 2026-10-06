"""perception.tracker - gives each detected person a stable ID across frames.

Config:
    camera: entrance
    backend: iou            "iou" (built in, no dependencies) or "bytetrack" (pip install supervision)
    labels: [person]        which detection labels to track
    iou_thresh: 0.3
    max_age_s: 1.0          keep a lost track this long before dropping it
    min_hits: 2             frames before a new track is published (filters flicker)

Listens:   det/<camera>
Publishes: track/<camera>

ID switches in crowds are the main failure mode for counting and dwell: if accuracy is poor,
try bytetrack, mount the entrance camera overhead, or raise the frame rate before touching
the analytics modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from drishti.core import Detections, Module, Track, Tracks
from drishti.core.geometry import iou


@dataclass
class _T:
    tid: int
    bbox: tuple[float, float, float, float]
    score: float
    last_ts: float
    hits: int = 1


class IouTracker:
    def __init__(self, iou_thresh: float = 0.3, max_age_s: float = 1.0, min_hits: int = 2):
        self.iou_thresh, self.max_age_s, self.min_hits = iou_thresh, max_age_s, min_hits
        self.tracks: list[_T] = []
        self.next_id = 1

    def update(self, boxes: list[tuple[tuple[float, float, float, float], float]], ts: float) -> list[_T]:
        pairs = sorted(
            ((iou(t.bbox, b), ti, bi) for ti, t in enumerate(self.tracks) for bi, (b, _) in enumerate(boxes)),
            reverse=True,
        )
        used_t: set[int] = set()
        used_b: set[int] = set()
        for score, ti, bi in pairs:
            if score < self.iou_thresh:
                break
            if ti in used_t or bi in used_b:
                continue
            t = self.tracks[ti]
            t.bbox, t.score, t.last_ts = boxes[bi][0], boxes[bi][1], ts
            t.hits += 1
            used_t.add(ti)
            used_b.add(bi)
        for bi, (b, s) in enumerate(boxes):
            if bi not in used_b:
                self.tracks.append(_T(self.next_id, b, s, ts))
                self.next_id += 1
        self.tracks = [t for t in self.tracks if ts - t.last_ts <= self.max_age_s]
        return [t for t in self.tracks if t.last_ts == ts and t.hits >= self.min_hits]


class Tracker(Module):
    subscribes = ("det/{camera}",)

    def setup(self) -> None:
        self.require("camera")
        self.labels = set(self.cfg.get("labels", ["person"]))
        self.kind = self.cfg.get("backend", "iou")
        if self.kind == "bytetrack":
            import supervision as sv  # optional dependency

            self.sv = sv
            self.bt = sv.ByteTrack(frame_rate=int(self.cfg.get("fps", 10)))
        else:
            self.iou_tracker = IouTracker(
                float(self.cfg.get("iou_thresh", 0.3)), float(self.cfg.get("max_age_s", 1.0)), int(self.cfg.get("min_hits", 2))
            )

    def on_message(self, topic: str, dets: Any) -> None:
        assert isinstance(dets, Detections)
        people = [d for d in dets.items if d.label in self.labels]
        if self.kind == "bytetrack":
            items = self._bytetrack(people)
        else:
            live = self.iou_tracker.update([(d.bbox, d.score) for d in people], dets.ts)
            items = [Track(t.tid, t.bbox, "person", t.score) for t in live]
        self.publish(f"track/{dets.camera_id}", Tracks(dets.camera_id, dets.ts, dets.seq, dets.frame_size, items))

    def _bytetrack(self, people: list[Any]) -> list[Track]:
        # TODO(owner: tracking): verify against the installed supervision version.
        import numpy as np

        sv = self.sv
        if not people:
            svd = sv.Detections.empty()
        else:
            svd = sv.Detections(
                xyxy=np.array([d.bbox for d in people], dtype=float),
                confidence=np.array([d.score for d in people], dtype=float),
                class_id=np.zeros(len(people), dtype=int),
            )
        out = self.bt.update_with_detections(svd)
        return [Track(int(out.tracker_id[i]), tuple(float(v) for v in out.xyxy[i]), "person") for i in range(len(out))]
