"""perception.detector - YOLO-family object detector (people, products, shelf labels, ...).

One module, many instances: the same code runs the person detector on the entrance camera
and the product detector on the shelf camera, each with its own model and labels.

Config:
    camera: entrance
    model: models/yolov8n.onnx          exported with: yolo export model=yolov8n.pt format=onnx
    backend: onnxruntime                see backends.py (litert_qnn for Qualcomm)
    providers: [CPUExecutionProvider]
    input_size: 640
    conf: 0.35
    iou: 0.5
    labels: {0: person}                 class index -> label; classes not listed are dropped
    min_interval_s: 0                   e.g. 60 for shelf cameras (one check a minute)

Listens:   frame/<camera>
Publishes: det/<camera>

Output format handled: YOLOv8 / YOLO11 ONNX export, shape (1, 4 + num_classes, N).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from drishti.core import Detection, Detections, Module


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize keeping aspect ratio and pad to size x size. Returns image, scale, (pad_x, pad_y)."""
    import cv2

    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - nw) / 2.0, (size - nh) / 2.0
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    out = cv2.copyMakeBorder(resized, top, size - nh - top, left, size - nw - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, scale, (float(left), float(top))


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Plain numpy non-maximum suppression. boxes: (N, 4) xyxy."""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        ious = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][ious <= iou_thr]
    return keep


def postprocess_yolo(
    out: np.ndarray, labels: dict[int, str], conf: float, iou_thr: float,
    scale: float, pad: tuple[float, float], frame_size: tuple[int, int],
) -> list[Detection]:
    """Turn a raw YOLOv8/11 output (1, 4+C, N) into detections in original-frame pixels."""
    pred = out[0].T if out.ndim == 3 else out.T  # (N, 4 + C)
    boxes_cxcywh, cls_scores = pred[:, :4], pred[:, 4:]
    cls_ids = cls_scores.argmax(axis=1)
    scores = cls_scores[np.arange(len(cls_ids)), cls_ids]
    wanted = np.isin(cls_ids, list(labels)) & (scores >= conf)
    if not wanted.any():
        return []
    b, s, c = boxes_cxcywh[wanted], scores[wanted], cls_ids[wanted]
    xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=1)
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad[0]) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad[1]) / scale
    w, h = frame_size
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
    dets: list[Detection] = []
    for cls in np.unique(c):
        idx = np.where(c == cls)[0]
        for k in nms(xyxy[idx], s[idx], iou_thr):
            i = idx[k]
            dets.append(Detection(tuple(float(v) for v in xyxy[i]), float(s[i]), labels[int(cls)]))
    return dets


class Detector(Module):
    subscribes = ("frame/{camera}",)

    def setup(self) -> None:
        self.require("camera", "model")
        from .backends import make_backend

        self.backend = make_backend(self.cfg)
        self.size = int(self.cfg.get("input_size", 640))
        self.labels = {int(k): v for k, v in (self.cfg.get("labels") or {0: "person"}).items()}
        self.conf = float(self.cfg.get("conf", 0.35))
        self.iou = float(self.cfg.get("iou", 0.5))
        self.min_interval = float(self.cfg.get("min_interval_s", 0))
        self.last_ts = -1e18

    def on_message(self, topic: str, frame: Any) -> None:
        if frame.ts - self.last_ts < self.min_interval:
            return
        self.last_ts = frame.ts
        img, scale, pad = letterbox(frame.image, self.size)
        blob = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        t0 = time.perf_counter()
        out = self.backend.infer(np.ascontiguousarray(blob))[0]
        infer_ms = (time.perf_counter() - t0) * 1000.0
        dets = postprocess_yolo(out, self.labels, self.conf, self.iou, scale, pad, frame.size)
        self.publish(
            f"det/{frame.camera_id}",
            Detections(frame.camera_id, frame.ts, frame.seq, frame.size, dets,
                       {"infer_ms": infer_ms, "model": self.cfg["model"], "labels": sorted(set(self.labels.values()))}),
        )
