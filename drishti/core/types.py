"""The data contracts. Every module speaks only these types.

Topics and what travels on them:

    frame/<camera>      Frame        raw image - LOCAL ONLY, never stored or sent off the box
    det/<camera>        Detections   boxes from a detector
    track/<camera>      Tracks       boxes with stable IDs
    event/<type>        Event        anything an analytics module concludes
    alert/<kind>        Alert        something a person should act on
    metric/<name>       Metric       health numbers (fps, latency, temperature)

Rule of thumb: frames stop at the perception layer; everything after it is metadata.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels of the source frame
Point = tuple[float, float]


@dataclass(slots=True)
class Frame:
    camera_id: str
    ts: float  # capture time, epoch seconds (replayed clips carry their own timeline)
    seq: int  # frame counter per camera
    image: Any  # numpy array, H x W x 3, BGR

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return (w, h)


@dataclass(slots=True)
class Detection:
    bbox: BBox
    score: float
    label: str  # "person", "product", "shelf_label", "vehicle", "plate", ...
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Detections:
    camera_id: str
    ts: float
    seq: int
    frame_size: tuple[int, int]  # (width, height)
    items: list[Detection]
    meta: dict[str, Any] = field(default_factory=dict)  # e.g. {"infer_ms": 12.3, "model": "yolov8n"}


@dataclass(slots=True)
class Track:
    track_id: int
    bbox: BBox
    label: str = "person"
    score: float = 1.0
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def anchor(self) -> Point:
        """Bottom-centre of the box: where a person's feet touch the floor."""
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)


@dataclass(slots=True)
class Tracks:
    camera_id: str
    ts: float
    seq: int
    frame_size: tuple[int, int]
    items: list[Track]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Event:
    type: str  # "entry", "exit", "dwell", "queue_update", "shelf_gap", ...
    camera_id: str
    ts: float
    data: dict[str, Any] = field(default_factory=dict)
    source: str = ""  # name of the module instance that produced it

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(slots=True)
class Alert:
    kind: str  # "restock", "open_counter", "planogram", ...
    message: str  # human-readable, shown on staff phones and the dashboard
    severity: str  # "info" | "warn" | "critical"
    ts: float
    camera_id: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(slots=True)
class Metric:
    name: str  # "fps", "infer_ms", "cpu_temp_c", ...
    value: float
    ts: float
    camera_id: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
