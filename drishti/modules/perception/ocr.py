"""perception.ocr - reads text from the image: shelf labels and price tags (retail),
number plates (the same module can serve an ANPR project).

Two ways to decide *where* to read:
    mode: regions      read fixed areas, e.g. each shelf row from the camera's `shelves:` config.
                       Needs no extra model - the OCR engine finds the text lines itself.
    mode: detections   read inside boxes from a detector (label: shelf_label / plate).

Config:
    camera: shelf_a
    mode: regions                  | detections
    target_label: shelf_label      (detections mode)
    engine: rapidocr               rapidocr (ONNX, fast on CPU) | tesseract
    parser: price                  price | plate | none
    min_conf: 0.5
    interval_s: 60                 labels do not change often; save compute

Listens:   frame/<camera>, det/<camera>
Publishes: event/text_read   {text, conf, bbox, parsed, row}
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from drishti.core import Detections, Frame, Module

# -- parsers: turn raw OCR text into fields ---------------------------------------------------

_PRICE = re.compile(r"(?:₹|rs\.?|inr|mrp)\s*[:\-]?\s*(\d{1,5}(?:[.,]\d{1,2})?)", re.IGNORECASE)
_PLATE = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{4})$")
_BH_PLATE = re.compile(r"^(\d{2})BH(\d{4})([A-Z]{1,2})$")


def parse_price(text: str) -> dict[str, Any]:
    m = _PRICE.search(text)
    if not m:
        return {}
    return {"price": float(m.group(1).replace(",", "."))}


def parse_indian_plate(text: str) -> dict[str, Any]:
    """Normalise OCR output to an Indian registration number, e.g. 'MH 12 AB 1234' -> MH12AB1234."""
    t = re.sub(r"[^A-Z0-9]", "", text.upper())
    m = _PLATE.match(t) or _BH_PLATE.match(t)
    return {"plate": t} if m else {}


PARSERS = {"price": parse_price, "plate": parse_indian_plate, "none": lambda _t: {}}


# -- engines: swap freely, same interface ------------------------------------------------------


class OcrEngine(Protocol):
    def read(self, image: Any) -> list[tuple[str, float, tuple[float, float, float, float]]]: ...


class RapidOcrEngine:
    """PaddleOCR models on ONNX Runtime. pip install rapidocr-onnxruntime (or rapidocr>=2)."""

    def __init__(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR  # 1.x package name

            self.new_api = False
        except ImportError:
            from rapidocr import RapidOCR  # 2.x package name

            self.new_api = True
        self.engine = RapidOCR()

    def read(self, image: Any) -> list[tuple[str, float, tuple[float, float, float, float]]]:
        res = self.engine(image)
        out = []
        if self.new_api:  # TODO(owner: OCR): verify field names against the installed version
            boxes, txts, scores = getattr(res, "boxes", None), getattr(res, "txts", None), getattr(res, "scores", None)
            items = zip(boxes if boxes is not None else [], txts or [], scores or [])
        else:
            items = ((r[0], r[1], r[2]) for r in (res[0] or []))
        for box, text, score in items:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            out.append((str(text), float(score), (min(xs), min(ys), max(xs), max(ys))))
        return out


class TesseractEngine:
    """Uses the tesseract binary via pytesseract. Slower and weaker on angled labels, but easy to install."""

    def __init__(self, psm: int = 6) -> None:
        import pytesseract

        self.pt = pytesseract
        self.config = f"--psm {psm}"

    def read(self, image: Any) -> list[tuple[str, float, tuple[float, float, float, float]]]:
        d = self.pt.image_to_data(image, config=self.config, output_type=self.pt.Output.DICT)
        lines: dict[tuple[int, int, int], list[int]] = {}
        for i, word in enumerate(d["text"]):
            if word.strip() and float(d["conf"][i]) >= 0:
                lines.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append(i)
        out = []
        for idx in lines.values():
            text = " ".join(d["text"][i] for i in idx)
            conf = sum(float(d["conf"][i]) for i in idx) / len(idx) / 100.0
            x1 = min(d["left"][i] for i in idx)
            y1 = min(d["top"][i] for i in idx)
            x2 = max(d["left"][i] + d["width"][i] for i in idx)
            y2 = max(d["top"][i] + d["height"][i] for i in idx)
            out.append((text, conf, (float(x1), float(y1), float(x2), float(y2))))
        return out


def make_engine(name: str) -> OcrEngine:
    if name == "rapidocr":
        return RapidOcrEngine()
    if name == "tesseract":
        return TesseractEngine()
    raise ValueError(f"unknown OCR engine '{name}'")


class OcrReader(Module):
    subscribes = ("frame/{camera}", "det/{camera}")

    def setup(self) -> None:
        self.require("camera")
        self.mode = self.cfg.get("mode", "regions")
        self.engine = make_engine(self.cfg.get("engine", "rapidocr"))
        self.parse = PARSERS[self.cfg.get("parser", "price")]
        self.min_conf = float(self.cfg.get("min_conf", 0.5))
        self.interval = float(self.cfg.get("interval_s", 60))
        self.target = self.cfg.get("target_label", "shelf_label")
        self.last_ts = -1e18
        self.frames: dict[int, Frame] = {}

    def on_message(self, topic: str, msg: Any) -> None:
        if isinstance(msg, Frame):
            if self.mode == "regions":
                if msg.ts - self.last_ts >= self.interval:
                    self.last_ts = msg.ts
                    self._read_regions(msg)
            else:
                self.frames[msg.seq] = msg
                for old in [s for s in self.frames if s < msg.seq - 5]:
                    del self.frames[old]
        elif isinstance(msg, Detections) and self.mode == "detections":
            frame = self.frames.get(msg.seq)
            if frame is None or msg.ts - self.last_ts < self.interval:
                return
            self.last_ts = msg.ts
            for d in msg.items:
                if d.label == self.target:
                    self._read_box(frame, d.bbox, row=None)

    def _read_regions(self, frame: Frame) -> None:
        rows = self.ctx.shelves(self.cfg["camera"])
        if not rows:
            w, h = frame.size
            self._read_box(frame, (0, 0, w, h), row=None)
        for row in rows:
            self._read_box(frame, tuple(row["rect"]), row=row["id"])

    def _read_box(self, frame: Frame, bbox: tuple, row: str | None) -> None:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        crop = frame.image[max(y1, 0):y2, max(x1, 0):x2]
        if crop.size == 0:
            return
        for text, conf, (bx1, by1, bx2, by2) in self.engine.read(crop):
            if conf < self.min_conf:
                continue
            box = (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)  # back to frame coordinates
            self.emit("text_read", frame.camera_id, frame.ts, text=text, conf=round(conf, 3), bbox=box,
                      parsed=self.parse(text), row=row)
