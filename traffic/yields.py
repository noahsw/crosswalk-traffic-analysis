"""Did drivers yield to pedestrians trying to cross?

For every pedestrian who actually crossed, we find when they started trying to cross
(waiting at the curb at one end of the crosswalk, or stepping in) and when they
finished. Waiting means being in the waiting area at that end, or standing still a little
further out (at the curb's edge, in the gutter, just beside the crosswalk): someone who stops
there means to cross, while someone walking past there may not. Each vehicle that reached the crosswalk in that window, in a lane the
pedestrian still had to cross, is classified as:

- yielded:            it stopped or nearly stopped (< stopped_mph) before the crosswalk,
                      and waited until the pedestrian no longer needed its lane.
                      Only the first such vehicle per direction counts, so a queue
                      behind it doesn't inflate the yield rate.
- did_not_yield:      it drove through without slowing below stopped_mph.
- too_close_to_stop:  it reached the crosswalk less than min_seconds_to_stop after the
                      pedestrian appeared, so the driver may not have been able to stop.

While a pedestrian waits at the curb, both lanes count (or, with waiting_counts_both_lanes
off, only the lane next to them). Once they are in the crosswalk, a lane counts until they
have passed its half of the road.

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
    others: list = field(default_factory=list)   # other pedestrians crossing together

    @property
    def crossings(self) -> list:
        return [self.crossing] + self.others


def _median_smooth(values: np.ndarray, k: int = 5) -> np.ndarray:
    if len(values) < 3:
        return values
    half = k // 2
    padded = np.pad(values, ((half, half), (0, 0)), mode="edge")
    return np.stack([np.median(padded[i:i + k], axis=0) for i in range(len(values))])


class YieldAnalyzer:
    MAX_GAP = 0.3    # seconds between detections that means the vehicle was hidden
    SETTLE_S = 0.5   # how long its smoothed position takes to catch up afterwards
    STILL_ZONE_FT = 6.0     # standing still this far into the road from a waiting area is waiting
    STILL_FT_PER_S = 1.0    # moving slower than this, over a second, is standing still

    def __init__(self, cfg: dict, cal: dict, direction_of):
        y = cfg["yielding"]
        self.min_lead = float(y["min_seconds_to_stop"])
        self.stopped_mph = float(y["stopped_mph"])
        self.lanes = dict(y.get("lanes") or {})
        self.waiting_both_lanes = bool(y.get("waiting_counts_both_lanes", True))
        self.direction_of = direction_of
        self.frame_w, self.frame_h = cal.get("image_size") or (1920, 1080)

        cw = np.asarray(cal["crosswalk"], float)
        self.crosswalk = cw.astype(np.float32)
        near_mid, far_mid = cw[:2].mean(0), cw[2:].mean(0)
        axis = far_mid - near_mid
        self.length_px = float(np.linalg.norm(axis))
        self.axis = axis / self.length_px
        self.origin = near_mid
        # A step past the curb line at each end: someone that far out has left the curb.
        self.margin = 0.03 * self.length_px
        # Approximate image scale near the crosswalk (pixels per foot).
        self.px_per_ft = self.length_px / float(cfg["road"]["crosswalk_length_ft"])
        # Line across the road through the middle of the crosswalk, a bit past each curb.
        self.pass_line = (tuple(near_mid - 0.25 * axis), tuple(far_mid + 0.25 * axis))
        depth = float(y["waiting_area_px"])
        self.wait_near = self._end_area(cw[0], cw[1], -self.axis, depth)
        self.wait_far = self._end_area(cw[3], cw[2], self.axis, depth)
        # The same areas reaching STILL_ZONE_FT into the road, for people standing still.
        slack = self.STILL_ZONE_FT * self.px_per_ft
        self.still_near = self._end_area(cw[0] + self.axis * slack, cw[1] + self.axis * slack,
                                         -self.axis, depth + slack)
        self.still_far = self._end_area(cw[3] - self.axis * slack, cw[2] - self.axis * slack,
                                        self.axis, depth + slack)

    @staticmethod
    def _end_area(a, b, outward, depth) -> np.ndarray:
        along = (b - a) / np.linalg.norm(b - a)
        widen = np.linalg.norm(b - a)   # a crosswalk's width to each side: people wait near, not on, it
        pts = [a - along * widen, b + along * widen,
               b + along * widen + outward * depth, a - along * widen + outward * depth]
        return np.asarray(pts, np.float32)

    def _proj(self, xy: np.ndarray) -> np.ndarray:
        return (xy - self.origin) @ self.axis

    def _still(self, t: np.ndarray, xy: np.ndarray) -> np.ndarray:
        """Whether they were standing still at each sample: over the second around it, they
        moved slower than STILL_FT_PER_S (and were seen for at least half of it)."""
        lo = np.searchsorted(t, t - 0.5)
        hi = np.searchsorted(t, t + 0.5, side="right") - 1
        span = t[hi] - t[lo]
        moved = np.linalg.norm(xy[hi] - xy[lo], axis=1)
        return (span >= 0.5) & (moved < self.STILL_FT_PER_S * self.px_per_ft * span)

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
        zone = self.still_near if start_end == "near" else self.still_far
        still = self._still(t, xy)
        # Walk back over the samples where they were at that curb, waiting.
        wait_i = near_i = first
        for i in range(first - 1, -1, -1):
            pt = (float(xy[i][0]), float(xy[i][1]))
            in_zone = cv2.pointPolygonTest(zone, pt, False) >= 0
            # At the curb: in the waiting area, inside the crosswalk but not yet in the road, or
            # standing still just beyond them.
            at_curb = (cv2.pointPolygonTest(area, pt, False) >= 0
                       or (inside[i] and not in_road[i]) or (in_zone and still[i]))
            # Up to 3 s away from it, or unseen, doesn't end the wait: someone waiting is often
            # hidden as they step in, or misplaced while a car passing in front hides their feet.
            # Nor does moving along just beyond it (walking along the gutter to the crosswalk),
            # though only time at the curb counts as waiting.
            if t[near_i] - t[i] > 3.0 or t[first] - t[i] > 120:
                break
            if at_curb:
                wait_i = i
            if at_curb or in_zone:
                near_i = i
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
        # A box cut off by the frame edge doesn't move with the vehicle, so skip those samples.
        clipped = {t for t, b in v.boxes if b[0] <= 3 or b[2] >= self.frame_w - 3 or b[3] >= self.frame_h - 3}
        # Nor does the smoothed position just after the vehicle was hidden (behind a pole, say):
        # the rolling median still holds positions from before, so it seems to stall.
        times = [t for t, _, _ in v.path]
        resumed = [t1 for t0, t1 in zip(times, times[1:]) if t1 - t0 > self.MAX_GAP]
        settling = {t for t in times for r in resumed if r <= t < r + self.SETTLE_S}
        pts = [(t, x, y) for t, x, y in v.path
               if t_from - window <= t <= t_to + 0.01 and t not in clipped and t not in settling]
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
            # Waiting at the curb: every lane, or only the one next to them.
            return (self.waiting_both_lanes or lane is None or lane == c.start_end), "waiting_at_curb"
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
        judged: dict[int, YieldEvent] = {}   # one judgement per vehicle, even for a group crossing
        for ped in sorted(peds, key=lambda p: p.first_pts):
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
                speed = float(np.median([x[1] for x in at])) if at else None
                lead = tv - c.wait_start
                if before:
                    min_speed = min(s for _, s in before)
                    # A yield if it stopped while the pedestrian was trying to cross and needed
                    # this lane, and then waited until they no longer did. (Stopping at the Havana
                    # stop sign and then pulling out in front of someone waiting to cross isn't
                    # yielding.)
                    stopped = [t for t, s in before
                               if s < self.stopped_mph and c.wait_start - 0.5 <= t <= c.exit + 0.5]
                    t_yield = next((t for t in stopped if self.conflicts(c, direction, t)[0]), None)
                elif speed is not None and speed >= self.stopped_mph:
                    # Seen too briefly on the way in to tell whether it stopped (it came into view
                    # just before the crosswalk, or the tracker lost it and picked it up again
                    # under a new id). Whatever it did before, it didn't yield if it's moving
                    # through the crosswalk while the pedestrian needs its lane: judged below.
                    min_speed, t_yield = None, None
                else:
                    continue       # not tracked long enough around the crosswalk to judge
                if t_yield is not None and (tv > c.exit or not self.conflicts(c, direction, tv)[0]):
                    if direction in yielded_dirs:
                        continue   # queued behind a yielding vehicle
                    outcome, phase = "yielded", self.conflicts(c, direction, t_yield)[1]
                else:
                    # Drove through, stopped only before the pedestrian needed its lane, or went
                    # while they still did: judged by when it reached the crosswalk.
                    if tv > c.exit:
                        continue   # arrived after the pedestrian had finished
                    ok, phase = self.conflicts(c, direction, tv)
                    if not ok:
                        continue
                    outcome = "too_close_to_stop" if lead < self.min_lead else "did_not_yield"
                if outcome == "yielded":
                    yielded_dirs.add(direction)
                if id(v) in judged:
                    # Same driver, another pedestrian in the same group: count the driver once.
                    judged[id(v)].others.append(c)
                    c.events.append(judged[id(v)])
                    continue
                e = YieldEvent(c, v, direction, tv, phase, lead, speed, min_speed, outcome)
                judged[id(v)] = e
                c.events.append(e)
        return crossings


def box_at(boxes: list, when: float, tolerance: float = 0.2):
    """The detection box nearest in time to `when`, or None."""
    if not boxes:
        return None
    ts = [b[0] for b in boxes]
    i = bisect_left(ts, when)
    best = min((k for k in (i - 1, i) if 0 <= k < len(boxes)), key=lambda k: abs(ts[k] - when))
    return boxes[best][1] if abs(ts[best] - when) <= tolerance else None
