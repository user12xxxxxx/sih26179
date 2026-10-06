"""analytics.shelf_gaps - turns product boxes into per-slot occupancy and empty gaps.

Pure geometry: it needs a product detector (perception.detector with a SKU-110K-style model,
label "product") on the shelf camera, plus the shelf layout drawn with tools/setup_zones.py:

    cameras:
      shelf_a:
        shelves:
          - id: A                      one shelf row
            rect: [100, 80, 1180, 260] x1, y1, x2, y2 in pixels
            slots:
              - {id: A1, x: [100, 380], product: "Maggi 70g"}
              - {id: A2, x: [380, 700], product: "Parle-G 250g"}

Config:
    camera: shelf_a
    product_label: product
    person_label: person          rows overlapped by a person are skipped (occlusion guard)
    min_gap_frac: 0.08            gaps narrower than this share of the row are ignored

Listens:   det/<camera>
Publishes: event/shelf_observation   {row, slots: [{id, product, occupancy}], gaps: [[x1, x2], ...]}
"""

from __future__ import annotations

from typing import Any

from drishti.core import Detections, Module
from drishti.core.geometry import boxes_overlap, covered_length, merge_intervals


def analyse_row(row: dict[str, Any], product_boxes: list[tuple[float, float, float, float]], min_gap_frac: float) -> dict[str, Any]:
    x1, y1, x2, y2 = row["rect"]
    spans = []
    for bx1, by1, bx2, by2 in product_boxes:
        cy = (by1 + by2) / 2
        if y1 <= cy <= y2 and bx2 > x1 and bx1 < x2:
            spans.append((max(bx1, x1), min(bx2, x2)))
    merged = merge_intervals(spans)
    gaps, cursor = [], x1
    for a, b in merged:
        if a - cursor >= min_gap_frac * (x2 - x1):
            gaps.append([round(cursor, 1), round(a, 1)])
        cursor = max(cursor, b)
    if x2 - cursor >= min_gap_frac * (x2 - x1):
        gaps.append([round(cursor, 1), round(x2, 1)])
    slots = []
    for s in row.get("slots", []):
        lo, hi = s["x"]
        occ = covered_length(merged, lo, hi) / max(hi - lo, 1e-6)
        slots.append({"id": s["id"], "product": s.get("product", ""), "occupancy": round(occ, 3)})
    return {"row": row["id"], "slots": slots, "gaps": gaps}


class ShelfGaps(Module):
    subscribes = ("det/{camera}",)

    def setup(self) -> None:
        self.require("camera")
        self.rows = self.ctx.shelves(self.cfg["camera"])
        if not self.rows:
            raise ValueError(f"no shelves defined for camera {self.cfg['camera']}")
        self.product = self.cfg.get("product_label", "product")
        self.person = self.cfg.get("person_label", "person")
        self.min_gap = float(self.cfg.get("min_gap_frac", 0.08))
        self.people_window = float(self.cfg.get("people_window_s", 2.0))
        self.last_people: tuple[float, list] = (-1e18, [])

    def on_message(self, topic: str, dets: Any) -> None:
        assert isinstance(dets, Detections)
        products = [d.bbox for d in dets.items if d.label == self.product]
        people = [d.bbox for d in dets.items if d.label == self.person]
        if people:
            # person boxes may come from a separate, faster detector on the same camera
            self.last_people = (dets.ts, people)
        # An empty product list is meaningful (an empty shelf!), so decide by which model sent it.
        if self.product not in dets.meta.get("labels", [self.product]):
            return
        if not people and dets.ts - self.last_people[0] <= self.people_window:
            people = self.last_people[1]
        for row in self.rows:
            if any(boxes_overlap(tuple(row["rect"]), p) for p in people):
                continue  # someone is standing in front of this row: don't judge it now
            self.emit("shelf_observation", dets.camera_id, dets.ts, **analyse_row(row, products, self.min_gap))
