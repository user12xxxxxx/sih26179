import shutil

import numpy as np
import pytest

from drishti.modules.perception.detector import postprocess_yolo
from drishti.modules.perception.ocr import parse_indian_plate, parse_price
from drishti.modules.perception.privacy import redact
from drishti.modules.perception.tracker import IouTracker


def test_yolo_postprocess_filters_nms_and_rescales():
    n_cls, cand = 80, 3
    out = np.zeros((1, 4 + n_cls, cand), dtype=np.float32)
    out[0, :4, 0] = [320, 320, 100, 200]   # person, strong
    out[0, 4 + 0, 0] = 0.9
    out[0, :4, 1] = [322, 318, 100, 200]   # same person, weaker -> removed by NMS
    out[0, 4 + 0, 1] = 0.6
    out[0, :4, 2] = [100, 100, 50, 50]     # a car -> not in labels
    out[0, 4 + 2, 2] = 0.95
    dets = postprocess_yolo(out, {0: "person"}, conf=0.3, iou_thr=0.5, scale=0.5, pad=(0.0, 0.0), frame_size=(1280, 1280))
    assert len(dets) == 1 and dets[0].label == "person"
    x1, y1, x2, y2 = dets[0].bbox
    assert (x1, y1, x2, y2) == (540.0, 440.0, 740.0, 840.0)  # 640-scale box mapped back to the full frame


def test_detector_module_end_to_end_with_a_fake_model():
    import drishti.modules.perception.backends as backends
    from drishti.core import Frame
    from drishti.modules.perception.detector import Detector
    from tests.helpers import make

    class FakeBackend:  # stands in for ONNX Runtime / QNN: one person in the middle
        def infer(self, blob):
            assert blob.shape == (1, 3, 640, 640)
            out = np.zeros((1, 84, 1), dtype=np.float32)
            out[0, :4, 0] = [320, 320, 64, 128]
            out[0, 4, 0] = 0.9
            return [out]

    original = backends.make_backend
    backends.make_backend = lambda cfg: FakeBackend()
    try:
        _, bus, rec = make(Detector, {"camera": "entrance", "model": "fake.onnx"})
        bus.publish("frame/entrance", Frame("entrance", 1.0, 0, np.zeros((720, 1280, 3), dtype=np.uint8)))
    finally:
        backends.make_backend = original
    dets = rec.of("det/entrance")[0]
    x1, y1, x2, y2 = dets.items[0].bbox  # 1280x720 is letterboxed to 640x360 + 140 px padding top/bottom
    assert (round(x1), round(y1), round(x2), round(y2)) == (576, 232, 704, 488)
    assert dets.meta["labels"] == ["person"]


def test_video_file_source_replays_a_clip(tmp_path):
    import threading

    import cv2

    from drishti.modules.sources.video_file import VideoFileSource
    from tests.helpers import make

    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    for i in range(20):
        writer.write(np.full((120, 160, 3), i * 10, dtype=np.uint8))
    writer.release()
    src, bus, rec = make(VideoFileSource, {"camera": "entrance", "path": str(path), "stride": 2,
                                          "start": "2026-10-01T17:00:00+05:30"})
    src.run(threading.Event())
    frames = rec.of("frame/entrance")
    assert len(frames) == 10 and abs((frames[1].ts - frames[0].ts) - 0.2) < 1e-6


def test_iou_tracker_keeps_ids_while_people_move():
    trk = IouTracker(min_hits=1)
    ids = []
    for step in range(5):
        live = trk.update([((100 + 5 * step, 100, 160 + 5 * step, 250), 0.9), ((600, 100, 660, 250), 0.9)], float(step))
        ids.append(sorted(t.tid for t in live))
    assert all(i == ids[0] for i in ids) and len(ids[0]) == 2


def test_text_parsers():
    assert parse_price("MRP ₹ 45.00 incl. taxes") == {"price": 45.0}
    assert parse_price("Rs.14") == {"price": 14.0}
    assert parse_price("Maggi Noodles") == {}
    assert parse_indian_plate("MH 12 AB 1234") == {"plate": "MH12AB1234"}
    assert parse_indian_plate("22 BH 1234 AA") == {"plate": "22BH1234AA"}
    assert parse_indian_plate("HELLO") == {}


def test_redaction_changes_only_person_boxes():
    img = (np.random.default_rng(0).random((200, 200, 3)) * 255).astype(np.uint8)
    out = redact(img, [(50, 50, 100, 100)], pad=0)
    assert not np.array_equal(out[50:100, 50:100], img[50:100, 50:100])
    assert np.array_equal(out[150:, 150:], img[150:, 150:])


def test_tesseract_reads_a_price_tag():
    pytest.importorskip("pytesseract")
    cv2 = pytest.importorskip("cv2")
    if not shutil.which("tesseract"):
        pytest.skip("tesseract binary not installed")
    from drishti.modules.perception.ocr import TesseractEngine

    img = np.full((120, 520, 3), 255, dtype=np.uint8)
    cv2.putText(img, "MRP Rs 45", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    texts = [t for t, _, _ in TesseractEngine().read(img)]
    assert any(parse_price(t).get("price") == 45.0 for t in texts), texts


def test_shelf_reference_occupancy_reads_full_and_empty_correctly():
    from drishti.modules.perception.shelf_reference import slot_occupancy

    full = np.full((10, 10), 200, dtype=np.uint8)
    empty = np.full((10, 10), 40, dtype=np.uint8)
    half = np.full((10, 10), 120, dtype=np.uint8)
    assert slot_occupancy(full, full, empty) == 1.0
    assert slot_occupancy(empty, full, empty) == 0.0
    assert abs(slot_occupancy(half, full, empty) - 0.5) < 1e-6
    assert slot_occupancy(full, full, full) == 1.0  # degenerate references: assume stocked


def test_shelf_reference_module_matches_frame_against_references(tmp_path):
    import cv2

    from drishti.core import Frame
    from drishti.modules.perception.shelf_reference import ShelfReference
    from tests.helpers import make

    full_img = np.full((100, 200), 200, dtype=np.uint8)
    empty_img = np.full((100, 200), 40, dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "full.jpg"), full_img)
    cv2.imwrite(str(tmp_path / "empty.jpg"), empty_img)

    now = np.zeros((100, 200, 3), dtype=np.uint8)
    now[:, :100] = 200  # A1 still looks fully stocked
    now[:, 100:] = 40   # A2 looks empty

    cams = {"shelf_a": {"shelves": [{"id": "A", "rect": [0, 0, 200, 100], "slots": [
        {"id": "A1", "x": [0, 100], "product": "Maggi"},
        {"id": "A2", "x": [100, 200], "product": "Parle-G"},
    ]}]}}
    _, bus, rec = make(ShelfReference, {"camera": "shelf_a", "full_ref": str(tmp_path / "full.jpg"),
                                        "empty_ref": str(tmp_path / "empty.jpg"), "interval_s": 0}, cams)
    bus.publish("frame/shelf_a", Frame("shelf_a", 10.0, 0, now))
    obs = rec.of("event/shelf_observation")[0].data
    occ = {s["id"]: s["occupancy"] for s in obs["slots"]}
    assert occ["A1"] == 1.0 and occ["A2"] == 0.0
    assert obs["gaps"] == [[100, 200]]


def test_ocr_module_reads_each_shelf_row():
    pytest.importorskip("pytesseract")
    cv2 = pytest.importorskip("cv2")
    if not shutil.which("tesseract"):
        pytest.skip("tesseract binary not installed")
    from drishti.core import Frame
    from drishti.modules.perception.ocr import OcrReader
    from tests.helpers import make

    img = np.full((400, 800, 3), 255, dtype=np.uint8)
    cv2.putText(img, "MRP Rs 14", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
    cams = {"shelf_a": {"shelves": [{"id": "A", "rect": [0, 50, 800, 200], "slots": []}]}}
    _, bus, rec = make(OcrReader, {"camera": "shelf_a", "engine": "tesseract", "min_conf": 0.3}, cams)
    bus.publish("frame/shelf_a", Frame("shelf_a", 100.0, 0, img))
    reads = rec.of("event/text_read")
    assert any(r.data["parsed"].get("price") == 14.0 and r.data["row"] == "A" for r in reads), [r.data for r in reads]
    x1, y1, x2, y2 = reads[0].data["bbox"]
    assert y1 >= 50  # box is in full-frame coordinates, inside the row
