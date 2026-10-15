"""Did drivers yield to pedestrians trying to cross?

For every pedestrian who actually crossed, we find when they started trying to cross
(waiting at the curb at one end of the crosswalk, or stepping in) and when they
finished. Each vehicle that reached the crosswalk in that window, in a lane the
pedestrian still had to cross, is classified as:

- yielded:            it stopped or nearly stopped (< stopped_mph) before the crosswalk.
                      Only the first such vehicle per direction counts, so a queue
                      behind it doesn't inflate the yield rate.
- did_not_yield:      it drove through without slowing below stopped_mph.
- too_close_to_stop:  it reached the crosswalk less than min_seconds_to_stop after the
                      pedestrian appeared, so the driver may not have been able to stop.

While a pedestrian waits at the curb, only the lane next to them counts. Once they are
in the crosswalk, a lane counts until they have passed its half of the road.

Speeds near the crosswalk are approximate: they use the crosswalk's measured length as
the image scale. They only need to tell "stopped" from "drove through".
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import segment_crossing
from .tracks import MPH_PER_FPS, PersonTrack, VehicleTrack


@dataclass
class Crossing:
    ped: PersonTrack
    wait_start: float     # when they started trying to cross (at the curb or stepping in)
    enter: float          # first step into the roadway
    exit: float           # last moment in the roadway
    start_end: str        # "near" or "far": the curb they started from
    direction: str
    events: list = field(default_factory=list)

    @property
    def wait_s(self) -> float:
        return self.enter - self.wait_start


@dataclass
class YieldEvent:
    crossing: Crossing
    vehicle: VehicleTrack
    direction: str
    passes: float          # when the vehicle reached the crosswalk
    phase: str             # "waiting_at_curb" or "in_crosswalk"
    lead_s: float          # seconds between the pedestrian appearing and the vehicle arriving
    speed_mph: float | None
    min_speed_mph: float | None
    outcome: str
    clip: str = ""


def _median_smooth(values: np.ndarray, k: int = 5) -> np.ndarray:
    if len(values) < 3:
        return values
    half = k // 2
    padded = np.pad(values, ((half, half), (0, 0)), mode="edge")
    return np.stack([np.median(padded[i:i + k], axis=0) for i in range(len(values))])


class YieldAnalyzer:
    def __init__(self, cfg: dict, cal: dict, direction_of):
        y = cfg["yielding"]
        self.min_lead = float(y["min_seconds_to_stop"])
        self.stopped_mph = float(y["stopped_mph"])
        self.lanes = dict(y.get("lanes") or {})
        self.direction_of = direction_of

        cw = np.asarray(cal["crosswalk"], float)
        self.crosswalk = cw.astype(np.float32)
        near_mid, far_mid = cw[:2].mean(0), cw[2:].mean(0)
        axis = far_mid - near_mid
        self.length_px = float(np.linalg.norm(axis))
        self.axis = axis / self.length_px
        self.origin = near_mid
        self.margin = 0.08 * self.length_px     # roughly the gutter at each end
        # Approximate image scale near the crosswalk (pixels per foot).
        self.px_per_ft = self.length_px / float(cfg["road"]["crosswalk_length_ft"])
        # Line across the road through the middle of the crosswalk, a bit past each curb.
        self.pass_line = (tuple(near_mid - 0.25 * axis), tuple(far_mid + 0.25 * axis))
        depth = float(y["waiting_area_px"])
        self.wait_near = self._end_area(cw[0], cw[1], -self.axis, depth)
        self.wait_far = self._end_area(cw[3], cw[2], self.axis, depth)

    @staticmethod
    def _end_area(a, b, outward, depth) -> np.ndarray:
        along = (b - a) / np.linalg.norm(b - a)
        widen = 0.5 * np.linalg.norm(b - a)
        pts = [a - along * widen, b + along * widen,
               b + along * widen + outward * depth, a - along * widen + outward * depth]
        return np.asarray(pts, np.float32)

    def _proj(self, xy: np.ndarray) -> np.ndarray:
        return (xy - self.origin) @ self.axis

    # ------------------------------------------------------------------ pedestrians
    def crossing_for(self, ped: PersonTrack) -> Crossing | None:
        if len(ped.path) < 3:
            return None
        t = np.asarray([p[0] for p in ped.path])
        xy = _median_smooth(np.asarray([(p[1], p[2]) for p in ped.path]))
        proj = self._proj(xy)
        inside = np.array([cv2.pointPolygonTest(self.crosswalk, (float(x), float(y)), True) >= -10
                           for x, y in xy])
        in_road = inside & (proj > self.margin) & (proj < self.length_px - self.margin)
        if not in_road.any():
            return None
        first = int(np.argmax(in_road))
        # The crossing continues while they stay in the road (allowing short dropouts).
        last = first
        for i in range(first + 1, len(t)):
            if in_road[i]:
                last = i
            elif t[i] - t[last] > 2.0:
                break
        start_end = "near" if proj[first] < self.length_px / 2 else "far"
        area = self.wait_near if start_end == "near" else self.wait_far
        # Walk back over the samples where they were at that curb, waiting.
        wait_i = first
        for i in range(first - 1, -1, -1):
            # At the curb: in the waiting area, or inside the crosswalk but not yet in the road.
            at_curb = (cv2.pointPolygonTest(area, (float(xy[i][0]), float(xy[i][1])), False) >= 0
                       or (inside[i] and not in_road[i]))
            if not at_curb or t[wait_i] - t[i] > 1.5 or t[first] - t[i] > 120:
                break
            wait_i = i
        return Crossing(ped, float(t[wait_i]), float(t[first]), float(t[last]), start_end,
                        ped.direction or "")

    def ped_proj_at(self, c: Crossing, when: float) -> float:
        path = c.ped.path
        i = min(range(len(path)), key=lambda k: abs(path[k][0] - when))
        return float(self._proj(np.asarray(path[i][1:])))

    # ------------------------------------------------------------------ vehicles
    def passes_at(self, v: VehicleTrack) -> float | None:
        for (t0, x0, y0), (t1, x1, y1) in zip(v.path, v.path[1:]):
            f = segment_crossing((x0, y0), (x1, y1), *self.pass_line)
            if f is not None:
                return t0 + f * (t1 - t0)
        return None

    def speed_mph(self, v: VehicleTrack, t_from: float, t_to: float,
                  window: float = 1.0) -> list[tuple[float, float]]:
        """Approximate (time, mph) over sliding `window`-second spans between t_from and t_to."""
        pts = [(t, x, y) for t, x, y in v.path if t_from - window <= t <= t_to + 0.01]
        out = []
        j = 0
        for i in range(len(pts)):
            while pts[i][0] - pts[j][0] > window:
                j += 1
            dt = pts[i][0] - pts[j][0]
            if dt >= 0.6 * window and pts[i][0] >= t_from:
                d = np.hypot(pts[i][1] - pts[j][1], pts[i][2] - pts[j][2]) / self.px_per_ft
                out.append((pts[i][0], d / dt * MPH_PER_FPS))
        return out

    def conflicts(self, c: Crossing, direction: str, when: float) -> tuple[bool, str]:
        """Does a vehicle travelling `direction` at time `when` conflict with this pedestrian?"""
        lane = self.lanes.get(direction)   # "near" or "far" half of the crosswalk
        if when < c.enter:
            # Waiting at the curb: only the lane next to them.
            return (lane is None or lane == c.start_end), "waiting_at_curb"
        p = self.ped_proj_at(c, when)
        half, buf = self.length_px / 2, 0.1 * self.length_px
        if lane is None:
            return True, "in_crosswalk"
        if c.start_end == "near":   # walking near -> far
            return (p < half + buf) if lane == "near" else True, "in_crosswalk"
        return (p > half - buf) if lane == "far" else True, "in_crosswalk"

    # ------------------------------------------------------------------ all together
    def analyze(self, peds: list[PersonTrack], vehicles: list[VehicleTrack]) -> list[Crossing]:
        passes = []
        for v in vehicles:
            if v.kind != "vehicle":
                continue
            tv = self.passes_at(v)
            if tv is not None:
                passes.append((tv, v))
        passes.sort(key=lambda x: x[0])
        times = [p[0] for p in passes]

        crossings = []
        for ped in peds:
            c = self.crossing_for(ped)
            if c is None:
                continue
            crossings.append(c)
            yielded_dirs = set()
            lo = bisect_left(times, c.wait_start)
            for tv, v in passes[lo:]:
                if tv > c.exit + 10:
                    break
                direction = self.direction_of(v)
                before = self.speed_mph(v, max(c.wait_start - 1, tv - 8), tv)
                at = self.speed_mph(v, tv - 0.5, tv + 0.5)
                if not before:
                    continue       # not tracked long enough before the crosswalk to judge
                t_slowest, min_speed = min(before, key=lambda x: x[1])
                speed = float(np.median([x[1] for x in at])) if at else None
                lead = tv - c.wait_start
                if min_speed < self.stopped_mph:
                    # It stopped: did the pedestrian need that lane at the moment it was stopped?
                    ok, phase = self.conflicts(c, direction, min(max(t_slowest, c.wait_start), c.exit))
                    if not ok or direction in yielded_dirs:
                        continue   # stopped for something else, or queued behind a yielding vehicle
                    outcome = "yielded"
                else:
                    if tv > c.exit:
                        continue   # arrived after the pedestrian had finished
                    ok, phase = self.conflicts(c, direction, tv)
                    if not ok:
                        continue
                    outcome = "too_close_to_stop" if lead < self.min_lead else "did_not_yield"
                if outcome == "yielded":
                    yielded_dirs.add(direction)
                c.events.append(YieldEvent(c, v, direction, tv, phase, lead, speed, min_speed, outcome))
        return crossings


def box_at(boxes: list, when: float, tolerance: float = 0.2):
    """The detection box nearest in time to `when`, or None."""
    if not boxes:
        return None
    ts = [b[0] for b in boxes]
    i = bisect_left(ts, when)
    best = min((k for k in (i - 1, i) if 0 <= k < len(boxes)), key=lambda k: abs(ts[k] - when))
    return boxes[best][1] if abs(ts[best] - when) <= tolerance else None
