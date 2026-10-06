"""Save one frame of a video with a numbered pixel grid, to read positions for lines and zones.

    python -m drishti snapshot clips/door.mp4            -> clips/door_grid.jpg
    python -m drishti snapshot clips/counter.mp4 --at 20 -> the frame 20 seconds in

The frame comes from the same video reader the pipeline uses, so the numbers you read off the
grid are exactly the pixel positions the pipeline sees (same size, same rotation).
"""

from __future__ import annotations

from pathlib import Path


def snapshot(video: str, at_s: float = 1.0, out: str | None = None, grid: int = 100) -> Path:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    cap.set(cv2.CAP_PROP_POS_MSEC, at_s * 1000.0)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read a frame {at_s} s into {video} - try a smaller --at")
    h, w = img.shape[:2]
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, max(0.5, min(w, h) / 1400)
    for x in range(0, w, grid):
        major = x % (grid * 5) == 0
        cv2.line(img, (x, 0), (x, h), (0, 255, 255) if major else (0, 200, 255), 2 if major else 1)
        for y_text in (int(28 * scale / 0.5), h - 10):
            cv2.putText(img, str(x), (x + 3, y_text), font, scale, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, str(x), (x + 3, y_text), font, scale, (0, 255, 255), 1, cv2.LINE_AA)
    for y in range(0, h, grid):
        major = y % (grid * 5) == 0
        cv2.line(img, (0, y), (w, y), (0, 255, 255) if major else (0, 200, 255), 2 if major else 1)
        for x_text in (3, w - int(70 * scale / 0.5)):
            cv2.putText(img, str(y), (x_text, y - 4), font, scale, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, str(y), (x_text, y - 4), font, scale, (0, 255, 255), 1, cv2.LINE_AA)
    path = Path(out) if out else Path(video).with_name(Path(video).stem + "_grid.jpg")
    cv2.imwrite(str(path), img)
    print(f"saved {path}  (video frame is {w} x {h} pixels; grid lines every {grid} px)")
    return path
