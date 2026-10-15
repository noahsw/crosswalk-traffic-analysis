"""Drawing helpers for the annotated video and calibration preview."""

from __future__ import annotations

import cv2
import numpy as np

# BGR colors
GREEN = (80, 200, 80)
YELLOW = (0, 210, 255)
RED = (60, 60, 230)
GRAY = (160, 160, 160)
WHITE = (255, 255, 255)
BLUE = (230, 160, 40)
MAGENTA = (220, 60, 220)
CYAN = (220, 220, 0)
ORANGE = (0, 140, 255)
BLACK = (0, 0, 0)

COUNT_LINE_COLOR = (0, 255, 255)
TRAP_COLOR = ORANGE
CROSSWALK_COLOR = (255, 120, 255)
GATE_COLOR = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def speed_color(mph: float, limit: float) -> tuple[int, int, int]:
    if mph <= limit:
        return GREEN
    if mph <= limit + 10:
        return YELLOW
    return RED


def put_label(img, text: str, org, color, scale: float = 0.6, thickness: int = 2,
              bg=BLACK, pad: int = 4, anchor: str = "bl") -> None:
    """Text with a filled background. anchor 'bl' = org is bottom-left, 'tl' = top-left."""
    (w, h), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = int(org[0]), int(org[1])
    if anchor == "tl":
        y += h + pad
    x = max(0, min(x, img.shape[1] - w - 2 * pad))
    y = max(h + pad, y)
    cv2.rectangle(img, (x, y - h - pad), (x + w + 2 * pad, y + base + pad // 2), bg, -1)
    cv2.putText(img, text, (x + pad, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_box(img, box, color, label: str | None = None, thickness: int = 2) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        put_label(img, label, (x1, y1 - 2), BLACK, scale=0.55, bg=color)


def _pts(seg):
    return np.asarray(seg, np.int32)


class SceneOverlay:
    """Static scene elements, pre-rendered once and blended onto each frame."""

    def __init__(self, cal: dict, shape, zones: bool = True, gates: bool = True,
                 alpha: float = 0.35):
        h, w = shape[:2]
        self.alpha = alpha
        self.layer = np.zeros((h, w, 3), np.uint8)
        mask = np.zeros((h, w), np.uint8)
        self.lines: list[tuple] = []   # crisp lines drawn on top, not blended
        if zones:
            cw = _pts(cal["crosswalk"])
            cv2.fillPoly(self.layer, [cw], CROSSWALK_COLOR)
            cv2.fillPoly(mask, [cw], 255)
            for seg, color, width in [(cal["count_line"], COUNT_LINE_COLOR, 8),
                                      (cal["speed_trap"]["line_a"], TRAP_COLOR, 6),
                                      (cal["speed_trap"]["line_b"], TRAP_COLOR, 6)]:
                p = _pts(seg)
                cv2.line(self.layer, tuple(p[0]), tuple(p[1]), color, width, cv2.LINE_AA)
                cv2.line(mask, tuple(p[0]), tuple(p[1]), 255, width, cv2.LINE_AA)
        if gates:
            for name, seg in cal["gates"].items():
                self.lines.append((_pts(seg), GATE_COLOR, 1, name))
        self.mask = mask > 0
        ys, xs = np.nonzero(self.mask)
        self.roi = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1) if len(ys) else None

    def apply(self, img) -> None:
        if self.roi is not None:
            y0, y1, x0, x1 = self.roi
            sub, lay, m = img[y0:y1, x0:x1], self.layer[y0:y1, x0:x1], self.mask[y0:y1, x0:x1]
            blended = cv2.addWeighted(sub, 1 - self.alpha, lay, self.alpha, 0)
            np.copyto(sub, blended, where=m[..., None])
        for p, color, width, name in self.lines:
            cv2.line(img, tuple(p[0]), tuple(p[1]), color, width, cv2.LINE_AA)
            cv2.putText(img, name, (int(p[0][0]) + 4, int(p[0][1]) - 6), FONT, 0.5, color, 1,
                        cv2.LINE_AA)


def draw_timestamp(img, text: str) -> None:
    put_label(img, text, (12, 12), WHITE, scale=1.0, thickness=2, anchor="tl", pad=8)


def draw_panel(img, lines: list[str], origin=(12, 70), scale: float = 0.7) -> None:
    """Semi-opaque panel of counter lines below the timestamp."""
    sizes = [cv2.getTextSize(t, FONT, scale, 2)[0] for t in lines]
    lh = max(s[1] for s in sizes) + 12
    w = max(s[0] for s in sizes) + 20
    x, y = origin
    h = lh * len(lines) + 10
    roi = img[y:y + h, x:x + w]
    cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0, dst=roi)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x + 10, y + lh * (i + 1)), FONT, scale, WHITE, 2, cv2.LINE_AA)


def draw_calibration_preview(img, cal: dict) -> np.ndarray:
    out = img.copy()
    SceneOverlay(cal, out.shape, alpha=0.45).apply(out)
    cw = cal["crosswalk"]
    put_label(out, "crosswalk", cw[3], BLACK, bg=CROSSWALK_COLOR, scale=0.5)
    put_label(out, "near side", cw[0], BLACK, bg=CROSSWALK_COLOR, scale=0.45)
    put_label(out, "count line", cal["count_line"][1], BLACK, bg=COUNT_LINE_COLOR, scale=0.5)
    trap = cal["speed_trap"]
    put_label(out, "trap A", trap["line_a"][0], BLACK, bg=TRAP_COLOR, scale=0.5)
    put_label(out, "trap B", trap["line_b"][0], BLACK, bg=TRAP_COLOR, scale=0.5)
    status = "measured" if trap.get("distance_measured") else "NOT MEASURED (guess)"
    put_label(out, f"Speed trap distance: {trap['distance_ft']:.1f} ft, {status}", (12, 12),
              WHITE, anchor="tl", scale=0.8)
    return out
