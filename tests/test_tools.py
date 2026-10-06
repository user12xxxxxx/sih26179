"""tools/setup_zones.py isn't a package (it's a standalone script), so it's loaded by file path.
Only the pure geometry-to-YAML helpers are tested here - the cv2 mouse/keyboard loop itself
needs a human clicking, same as any GUI tool."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("setup_zones", ROOT / "tools" / "setup_zones.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_line_keeps_click_order():
    m = _load()
    assert m.build_line([[10, 20], [30, 20], [20, 5]]) == {"a": [10, 20], "b": [30, 20], "in_point": [20, 5]}


def test_build_line_guesses_inside_from_two_points():
    from drishti.core.geometry import side_of_line
    m = _load()
    a, b = [380, 400], [900, 400]           # a door line across the top of a 1280x720 frame
    ln = m.build_line([a, b], (1280, 720))
    assert ln["a"] == a and ln["b"] == b
    # the guess must be usable: line_counter rejects an in_point that sits on the line
    assert side_of_line(ln["in_point"], a, b) != 0
    # and it must land on the same side as the frame centre (below the line here)
    assert side_of_line(ln["in_point"], a, b) == side_of_line([640, 360], a, b)


def test_build_line_guess_survives_a_line_through_the_frame_centre():
    from drishti.core.geometry import side_of_line
    m = _load()
    a, b = [0, 360], [1280, 360]            # passes exactly through the centre of a 1280x720 frame
    ln = m.build_line([a, b], (1280, 720))
    assert side_of_line(ln["in_point"], a, b) != 0


def test_build_zone_returns_a_copy_not_the_live_click_list():
    m = _load()
    pts = [[1, 1], [2, 1], [2, 2], [1, 2]]
    zone = m.build_zone(pts)
    assert zone == pts
    pts.append([9, 9])
    assert zone == [[1, 1], [2, 1], [2, 2], [1, 2]]  # unaffected by further clicks


def test_build_shelf_row_splits_slots_at_cuts_regardless_of_click_order():
    m = _load()
    row = m.build_shelf_row("A", [[100, 300], [900, 0], [500, 999]])  # bottom-left, top-right, one cut
    assert row["rect"] == [100, 0, 900, 300]
    assert [s["x"] for s in row["slots"]] == [[100, 500], [500, 900]]
    assert [s["id"] for s in row["slots"]] == ["A1", "A2"]


def test_build_shelf_row_with_no_cuts_is_one_slot():
    m = _load()
    row = m.build_shelf_row("B", [[0, 0], [400, 200]])
    assert [s["x"] for s in row["slots"]] == [[0, 400]]
