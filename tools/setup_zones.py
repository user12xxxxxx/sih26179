"""Draw a camera's lines, zones and shelf rows on a snapshot, save them as YAML.

    python tools/setup_zones.py --image snapshots/entrance.jpg --out configs/cameras/entrance.yaml

Everything happens in the image window - click it once first so it has keyboard focus.

    l   new counting line: click 2 points across the doorway, then Enter. Optionally click a
        3rd point INSIDE the store first - without it the inside side is guessed (image centre)
    z   new zone: click the corners, then Enter
    r   new shelf row: click 2 opposite corners, then slot boundaries left->right, Enter
    u   undo the last shape
    w   write the YAML
    q   quit

After Enter you type the shape's name in the window itself (Enter accepts, Esc cancels).
Needs OpenCV with GUI support (opencv-python, not the -headless wheel; on NixOS use
python3Packages.opencv4 with GTK enabled).
TODO(owner: setup tool): a browser version served by `drishti serve` would avoid GUI quirks.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from drishti.core.geometry import side_of_line  # same function line_counter reads in_point with

ESC, BACKSPACE, NO_KEY = 27, 8, 255
ENTER = (13, 10)


def guess_in_point(a: list[int], b: list[int], frame_size: tuple[int, int] | None = None) -> list[int]:
    """Pick a point on the 'inside' side of a->b when the user did not click one.

    The image centre is the best guess: an entrance camera is normally mounted inside looking
    at the door, so most of the frame is the shop floor. If the line happens to pass through
    the centre, step perpendicular from the line's midpoint instead.
    """
    if frame_size:
        centre = [frame_size[0] // 2, frame_size[1] // 2]
        if side_of_line(centre, a, b) != 0:
            return centre
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = max((dx * dx + dy * dy) ** 0.5, 1e-9)
    step = (max(frame_size) * 0.1) if frame_size else (n * 0.5)
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return [int(mid[0] - dy / n * step), int(mid[1] + dx / n * step)]


def build_line(pts: list[list[int]], frame_size: tuple[int, int] | None = None) -> dict:
    """2 points define the line; an optional 3rd marks the inside. Without it, it is guessed."""
    a, b = pts[0], pts[1]
    in_point = pts[2] if len(pts) >= 3 else guess_in_point(a, b, frame_size)
    return {"a": a, "b": b, "in_point": in_point}


def build_zone(pts: list[list[int]]) -> list[list[int]]:
    return pts[:]


def build_shelf_row(row_id: str, pts: list[list[int]]) -> dict:
    (x1, y1), (x2, y2) = pts[0], pts[1]
    cuts = sorted([min(x1, x2)] + [p[0] for p in pts[2:]] + [max(x1, x2)])
    slots = [{"id": f"{row_id}{i + 1}", "x": [cuts[i], cuts[i + 1]], "product": ""} for i in range(len(cuts) - 1)]
    return {"id": row_id, "rect": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], "slots": slots}


#: points needed before a shape can be named and saved
NEEDED = {"line": 2, "zone": 3, "row": 2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    h, w = img.shape[:2]
    out = Path(args.out)
    data = (yaml.safe_load(out.read_text()) if out.exists() else None) or {}
    data.setdefault("frame_size", [w, h])
    data.setdefault("lines", {})
    data.setdefault("zones", {})
    data.setdefault("shelves", [])

    history: list[tuple[str, str]] = []
    state = {"mode": None, "pts": [], "naming": None, "buf": "", "flash": "", "flash_until": 0.0}

    def flash(msg: str) -> None:
        """Feedback goes in the window: nobody is looking at the terminal while drawing."""
        state["flash"], state["flash_until"] = msg, time.monotonic() + 3.0
        print(msg)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and state["mode"] and not state["naming"]:
            state["pts"].append([int(x), int(y)])

    def default_name(kind: str) -> str:
        if kind == "line":
            return f"line{len(data['lines']) + 1}"
        if kind == "zone":
            return f"zone{len(data['zones']) + 1}"
        return chr(ord("A") + len(data["shelves"]))

    def start_naming() -> None:
        """Enough points collected: ask for a name *in the window* (never block on input())."""
        kind, pts = state["mode"], state["pts"]
        if not kind or len(pts) < NEEDED[kind]:
            return  # Enter pressed too early - keep collecting clicks
        state["naming"], state["buf"] = kind, ""

    def commit_name() -> None:
        kind, pts = state["naming"], state["pts"]
        name = state["buf"].strip() or default_name(kind)
        if kind == "line":
            data["lines"][name] = build_line(pts, (w, h))
            history.append(("lines", name))
            if len(pts) < 3:
                flash(f"'{name}' saved - inside was guessed (green dot), press u to redo with a 3rd click")
        elif kind == "zone":
            data["zones"][name] = build_zone(pts)
            history.append(("zones", name))
        else:
            data["shelves"].append(build_shelf_row(name, pts))
            history.append(("shelves", name))
        state.update(mode=None, pts=[], naming=None, buf="")

    def undo() -> None:
        if not history:
            flash("nothing to undo")
            return
        kind, name = history.pop()
        if kind == "shelves":
            data["shelves"] = [r for r in data["shelves"] if r["id"] != name]
        else:
            data[kind].pop(name, None)
        flash(f"removed {name}")

    def write() -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(data, sort_keys=False))
        flash(f"saved {out.resolve()}")

    cv2.namedWindow("setup", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("setup", min(w, 1280), min(h, 720) + 40)
    cv2.setMouseCallback("setup", on_mouse)
    print("Click the image window once so it has keyboard focus, then use l / z / r / u / w / q.")

    while True:
        if cv2.getWindowProperty("setup", cv2.WND_PROP_VISIBLE) < 1:
            break  # closed with the window's own [x] button
        canvas = img.copy()
        for name, ln in data["lines"].items():
            cv2.line(canvas, tuple(ln["a"]), tuple(ln["b"]), (0, 200, 255), 2)
            cv2.circle(canvas, tuple(ln["in_point"]), 5, (0, 255, 0), -1)
            cv2.putText(canvas, name, tuple(ln["a"]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        for name, poly in data["zones"].items():
            cv2.polylines(canvas, [np.array(poly, dtype=np.int32).reshape(-1, 1, 2)], True, (255, 120, 0), 2)
            cv2.putText(canvas, name, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)
        for row in data["shelves"]:
            x1, y1, x2, y2 = row["rect"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 220, 80), 2)
            for s in row["slots"]:
                cv2.line(canvas, (s["x"][1], y1), (s["x"][1], y2), (80, 220, 80), 1)
                cv2.putText(canvas, s["id"], (s["x"][0] + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 1)
        for p in state["pts"]:
            cv2.circle(canvas, tuple(p), 4, (0, 0, 255), -1)

        have = f"{len(data['lines'])} line(s), {len(data['zones'])} zone(s), {len(data['shelves'])} row(s)"
        if state["naming"]:
            status = f"name this {state['naming']}: {state['buf']}_   (Enter=ok, default '{default_name(state['naming'])}', Esc=cancel)"
        elif state["mode"] == "line":
            extra = "Enter to finish (inside guessed), or click a 3rd point inside the store"
            status = f"line: {len(state['pts'])}/2 points - " + (extra if len(state["pts"]) >= 2 else "click across the doorway")
        elif state["mode"]:
            need = NEEDED[state["mode"]]
            status = f"{state['mode']}: {len(state['pts'])} point(s), need {need}+ then Enter   [Esc]cancel"
        elif time.monotonic() < state["flash_until"]:
            status = state["flash"]
        else:
            status = f"[l]ine  [z]one  shelf [r]ow  [u]ndo  [w]rite  [q]uit   -   {have}"
        cv2.rectangle(canvas, (0, h - 34), (w, h), (0, 0, 0), -1)
        cv2.putText(canvas, status, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("setup", canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == NO_KEY:
            continue

        if state["naming"]:  # typing a name: every other shortcut is just a character
            if key in ENTER:
                commit_name()
            elif key == ESC:
                state.update(naming=None, buf="")
            elif key == BACKSPACE:
                state["buf"] = state["buf"][:-1]
            elif 32 <= key <= 126:
                state["buf"] += chr(key)
            continue

        cmd = chr(key).lower() if 32 <= key <= 126 else ""  # so Caps Lock / Shift still work
        if cmd == "q":
            break
        if cmd in ("l", "z", "r"):
            state.update(mode={"l": "line", "z": "zone", "r": "row"}[cmd], pts=[])
        elif key == ESC:
            state.update(mode=None, pts=[])
        elif key in ENTER:
            start_naming()
        elif cmd == "u":
            undo()
        elif cmd == "w":
            write()
        if state["mode"] == "line" and len(state["pts"]) == 3:
            start_naming()  # a line is always exactly 3 clicks

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
