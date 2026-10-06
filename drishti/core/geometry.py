"""Small, dependency-free geometry helpers shared by analytics modules."""

from __future__ import annotations

from typing import Sequence

from .types import BBox, Point


def cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def side_of_line(p: Point, a: Point, b: Point) -> int:
    """+1 / -1 depending on which side of the infinite line a->b the point lies, 0 if on it."""
    c = cross(a, b, p)
    return (c > 0) - (c < 0)


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """True if segment p1-p2 crosses segment q1-q2 (touching counts)."""
    d1 = cross(q1, q2, p1)
    d2 = cross(q1, q2, p2)
    d3 = cross(p1, p2, q1)
    d4 = cross(p1, p2, q2)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 != 0 and d2 != 0 and d3 != 0 and d4 != 0:
        return True

    def on_seg(o: Point, a: Point, b: Point) -> bool:
        return min(o[0], b[0]) <= a[0] <= max(o[0], b[0]) and min(o[1], b[1]) <= a[1] <= max(o[1], b[1])

    return (
        (d1 == 0 and on_seg(q1, p1, q2))
        or (d2 == 0 and on_seg(q1, p2, q2))
        or (d3 == 0 and on_seg(p1, q1, p2))
        or (d4 == 0 and on_seg(p1, q2, p2))
    )


def point_in_polygon(p: Point, poly: Sequence[Point]) -> bool:
    """Ray casting. Works for any simple polygon given as a list of (x, y)."""
    x, y = p
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def boxes_overlap(a: BBox, b: BBox) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def covered_length(intervals: Sequence[tuple[float, float]], lo: float, hi: float) -> float:
    total = 0.0
    for a, b in merge_intervals(intervals):
        total += max(0.0, min(b, hi) - max(a, lo))
    return total


class Homography:
    """Maps camera pixels to floor-plan coordinates from 4 point pairs (for store-wide heatmaps)."""

    def __init__(self, src: Sequence[Point], dst: Sequence[Point]):
        import numpy as np

        if len(src) != 4 or len(dst) != 4:
            raise ValueError("homography needs exactly 4 source and 4 destination points")
        rows = []
        rhs = []
        for (x, y), (u, v) in zip(src, dst):
            rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
            rhs.append(u)
            rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
            rhs.append(v)
        h = np.linalg.solve(np.array(rows, dtype=float), np.array(rhs, dtype=float))
        self.m = np.append(h, 1.0).reshape(3, 3)

    def __call__(self, p: Point) -> Point:
        x, y = p
        m = self.m
        w = m[2, 0] * x + m[2, 1] * y + m[2, 2]
        return ((m[0, 0] * x + m[0, 1] * y + m[0, 2]) / w, (m[1, 0] * x + m[1, 1] * y + m[1, 2]) / w)
