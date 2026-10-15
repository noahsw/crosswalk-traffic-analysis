"""Small 2D geometry helpers in image pixel coordinates."""

from __future__ import annotations

import numpy as np

Point = tuple[float, float]


def _cross(a, b) -> float:
    return a[0] * b[1] - a[1] * b[0]


def segment_crossing(p: Point, q: Point, a: Point, b: Point) -> float | None:
    """If motion p->q crosses segment a-b, return the fraction t in [0, 1] along p->q."""
    p, q, a, b = (np.asarray(v, float) for v in (p, q, a, b))
    r, s = q - p, b - a
    denom = _cross(r, s)
    if abs(denom) < 1e-9:
        return None
    t = _cross(a - p, s) / denom
    u = _cross(a - p, r) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return float(t)
    return None


def point_in_polygon(pt: Point, poly) -> bool:
    import cv2
    return cv2.pointPolygonTest(np.asarray(poly, np.float32), (float(pt[0]), float(pt[1])), False) >= 0


def containment(inner, outer) -> float:
    """Fraction of box `inner` (x1,y1,x2,y2) that lies inside box `outer`."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


COMPASS = {"east": (1, 0), "west": (-1, 0), "north": (0, 1), "south": (0, -1)}


def classify_movement(entry: str, exit: str) -> str:
    """through / left_turn / right_turn / u_turn from compass-named gates.

    Entering via the west gate means heading east, and so on.
    """
    if entry not in COMPASS or exit not in COMPASS:
        return "unknown"
    hin = tuple(-c for c in COMPASS[entry])
    hout = COMPASS[exit]
    dot = hin[0] * hout[0] + hin[1] * hout[1]
    if dot > 0:
        return "through"
    if dot < 0:
        return "u_turn"
    return "left_turn" if _cross(hin, hout) > 0 else "right_turn"
