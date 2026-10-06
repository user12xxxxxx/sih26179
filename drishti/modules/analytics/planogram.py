"""analytics.planogram - checks that the right product (and price) sits in each slot.

Uses text read from shelf-edge labels (perception.ocr) and the expected layout from the
camera's `shelves:` config (each slot's `product` and optional `price`).

Config:
    camera: shelf_a
    min_similarity: 0.6      fuzzy match between label text and expected product name
    check_price: true

Listens:   event/text_read
Publishes: event/planogram_mismatch   {slot, expected, read, similarity}
           event/price_mismatch       {slot, expected_price, read_price}

TODO(owner: shelf): a stronger version compares product *appearance* against a reference
photo of each slot (embedding similarity), which also works where labels are missing.
"""

from __future__ import annotations

import difflib
from typing import Any

from drishti.core import Event, Module


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


class PlanogramCheck(Module):
    subscribes = ("event/text_read",)

    def setup(self) -> None:
        self.require("camera")
        self.rows = self.ctx.shelves(self.cfg["camera"])
        self.min_sim = float(self.cfg.get("min_similarity", 0.6))
        self.check_price = bool(self.cfg.get("check_price", True))

    def slot_at(self, x: float, y: float) -> dict[str, Any] | None:
        for row in self.rows:
            x1, y1, x2, y2 = row["rect"]
            # labels sit on the shelf edge, just below the row: allow some slack underneath
            if x1 <= x <= x2 and y1 <= y <= y2 + 0.5 * (y2 - y1):
                for s in row.get("slots", []):
                    if s["x"][0] <= x < s["x"][1]:
                        return s
        return None

    def on_message(self, topic: str, ev: Any) -> None:
        assert isinstance(ev, Event)
        if ev.camera_id != self.cfg["camera"]:
            return
        x1, y1, x2, y2 = ev.data["bbox"]
        slot = self.slot_at((x1 + x2) / 2, (y1 + y2) / 2)
        if not slot:
            return
        text = ev.data["text"]
        expected = slot.get("product", "")
        price = (ev.data.get("parsed") or {}).get("price")
        if expected and price is None:  # a name label, not a price tag
            sim = similarity(text, expected)
            if sim < self.min_sim:
                self.emit("planogram_mismatch", ev.camera_id, ev.ts, slot=slot["id"], expected=expected, read=text,
                          similarity=round(sim, 2))
        if self.check_price and price is not None and slot.get("price") is not None and abs(price - float(slot["price"])) > 0.01:
            self.emit("price_mismatch", ev.camera_id, ev.ts, slot=slot["id"], expected_price=float(slot["price"]), read_price=price)
