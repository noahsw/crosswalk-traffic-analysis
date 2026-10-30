"""Per-track measurements: counting, movement, speed trap and pedestrian crossings.

Everything here is causal (uses only past samples) so the live overlay and the
CSV outputs always agree.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import (COMPASS, classify_movement, containment, iou,
                       segment_crossing)

MPH_PER_FPS = 3600 / 5280  # feet/second -> miles/hour


def _ground_point(box) -> tuple[float, float]:
    """Bottom-center of a box: where the object touches the road."""
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2.0, float(y2))


@dataclass
class VehicleTrack:
    tid: int
    kind: str  # "vehicle" or "bicycle"
    smoothing: int
    first_pts: float = 0.0
    last_pts: float = 0.0
    last_box: tuple = ()
    class_votes: Counter = field(default_factory=Counter)
    raw: deque = field(default_factory=deque)
    path: list = field(default_factory=list)          # (pts, x, y) smoothed ground points
    boxes: list = field(default_factory=list)         # (pts, box) raw detections
    gate_events: list = field(default_factory=list)   # (pts, gate, "in"|"out")
    count_pts: float | None = None
    count_dir: str | None = None
    trap_a: float | None = None
    trap_b: float | None = None
    snapshot: np.ndarray | None = None
    snapshot_box: tuple = ()
    movement: str = "unknown"
    entry: str | None = None
    exit: str | None = None
    inferred: bool = False
    speed_mph: float | None = None
    n_trap_points: int = 0

    @property
    def cls(self) -> str:
        return self.class_votes.most_common(1)[0][0] if self.class_votes else self.kind

    @property
    def relevant(self) -> bool:
        """Crossed something we measure; parked or background objects are dropped."""
        return bool(self.gate_events) or self.count_pts is not None or self.trap_a is not None


@dataclass
class PersonTrack:
    tid: int
    first_pts: float = 0.0
    last_pts: float = 0.0
    prev_inside_pts: float | None = None
    inside_seconds: float = 0.0
    first_inside_pts: float | None = None
    first_proj: float | None = None
    min_proj: float = float("inf")
    max_proj: float = float("-inf")
    last_proj: float = 0.0
    samples: int = 0
    rider_samples: int = 0
    crossed_pts: float | None = None
    direction: str | None = None
    closest_px: float = float("-inf")   # signed distance to the crosswalk (positive = inside)
    path: list = field(default_factory=list)    # (pts, x, y) feet
    boxes: list = field(default_factory=list)   # (pts, box)
    snapshot: np.ndarray | None = None
    snapshot_box: tuple = ()

    def reason(self, min_seconds: float, min_travel: float) -> str:
        """Why this person was or wasn't counted as a crossing."""
        if self.crossed_pts is not None:
            return "counted"
        if self.first_inside_pts is None:
            return "near_but_never_inside"
        if self.rider_samples > self.samples / 2:
            return "riding_a_bike_or_motorcycle"
        if self.max_proj - self.min_proj < min_travel:
            return "did_not_move_far_enough_across"
        if self.inside_seconds < min_seconds:
            return "too_briefly_inside"
        return "other"


OPPOSITE_LEG = {"east": "west", "west": "east", "north": "south", "south": "north"}
PED_NEAR_PX = 60  # people whose feet come this close to the crosswalk are logged for review


class Analyzer:
    def __init__(self, cfg: dict, cal: dict, start_time, on_snapshot_needed=None,
                 on_ped_snapshot=None):
        self.cfg, self.cal, self.start_time = cfg, cal, start_time
        t = cfg["tracking"]
        self.smoothing = int(t["smoothing_samples"])
        self.finalize_after = float(t["finalize_after_seconds"])
        self.limit = float(cfg["speed_limit_mph"])
        self.dir_labels = cfg["direction_labels"]
        self.cw_labels = cfg["crosswalk_direction_labels"]

        self.count_line = [tuple(p) for p in cal["count_line"]]
        self.count_leg = cal.get("count_line_leg")
        trap = cal["speed_trap"]
        self.trap_a = [tuple(p) for p in trap["line_a"]]
        self.trap_b = [tuple(p) for p in trap["line_b"]]
        self.trap_ft = float(trap["distance_ft"])
        self.gates = {name: [tuple(p) for p in seg] for name, seg in cal["gates"].items()}

        cw = np.asarray(cal["crosswalk"], float)
        self.crosswalk = cw
        near_mid, far_mid = cw[:2].mean(0), cw[2:].mean(0)
        axis = far_mid - near_mid
        self.cw_origin = near_mid
        self.cw_axis_len = float(np.linalg.norm(axis))
        self.cw_axis = axis / self.cw_axis_len
        # Gate "inside" side = the side the crosswalk/scene centre is on.
        centre = cw.mean(0)
        self.gate_inside_sign = {n: np.sign(self._side(seg, centre)) for n, seg in self.gates.items()}

        self.vehicles: dict[tuple[str, int], VehicleTrack] = {}
        self.people: dict[int, PersonTrack] = {}
        self.ped_alias: dict[int, int] = {}   # track id -> the earlier id of the same person
        self.vehicle_records: list[VehicleTrack] = []
        self.ped_records: list[PersonTrack] = []
        self.ped_reviewed: list[tuple[PersonTrack, str]] = []   # everyone who came near
        self.on_ped_snapshot = on_ped_snapshot
        self.ped_snapshots_saved: Counter = Counter()
        self.ignored_tracks = 0
        self.snapshot_quota = int(cfg["output"]["movement_check_samples"])
        self.snapshots_saved: Counter = Counter()
        self.on_finalize_snapshot = on_snapshot_needed

        # Live counters for the overlay (always equal to what lands in the CSVs).
        self.counts = Counter()

    @staticmethod
    def _side(seg, pt) -> float:
        (ax, ay), (bx, by) = seg
        return (bx - ax) * (pt[1] - ay) - (by - ay) * (pt[0] - ax)

    # ------------------------------------------------------------------ vehicles
    def update_vehicles(self, pts: float, frame, boxes, ids, classes, kind: str) -> None:
        for box, tid, cls in zip(boxes, ids, classes):
            key = (kind, int(tid))
            tr = self.vehicles.get(key)
            if tr is None:
                tr = self.vehicles[key] = VehicleTrack(int(tid), kind, self.smoothing, first_pts=pts)
            tr.last_pts, tr.last_box = pts, tuple(float(v) for v in box)
            tr.boxes.append((pts, tr.last_box))
            tr.class_votes[cls] += 1
            tr.raw.append(_ground_point(box))
            while len(tr.raw) > self.smoothing:
                tr.raw.popleft()
            x, y = np.median(np.asarray(tr.raw), axis=0)
            prev = tr.path[-1] if tr.path else None
            tr.path.append((pts, float(x), float(y)))
            if prev is not None:
                self._check_crossings(tr, prev, tr.path[-1], frame)

    def _check_crossings(self, tr: VehicleTrack, prev, cur, frame) -> None:
        p, q = prev[1:], cur[1:]
        dt = cur[0] - prev[0]

        for name, seg in self.gates.items():
            if segment_crossing(p, q, *seg) is not None:
                toward_inside = np.sign(self._side(seg, q)) == self.gate_inside_sign[name]
                tr.gate_events.append((cur[0], name, "in" if toward_inside else "out"))
                self._resolve_movement(tr)
                if tr.snapshot is None and self._want_snapshot():
                    tr.snapshot, tr.snapshot_box = frame.copy(), tr.last_box

        if tr.count_pts is None:
            t = segment_crossing(p, q, *self.count_line)
            if t is not None:
                tr.count_pts = prev[0] + t * dt
                key = "left_to_right" if q[0] >= p[0] else "right_to_left"
                tr.count_dir = self.dir_labels.get(key, key)
                self.counts[f"{tr.kind}_total"] += 1
                self.counts[f"{tr.kind}_{tr.count_dir}"] += 1
                self._resolve_movement(tr)
                if self._want_snapshot():
                    tr.snapshot, tr.snapshot_box = frame.copy(), tr.last_box

        for attr, seg in (("trap_a", self.trap_a), ("trap_b", self.trap_b)):
            if getattr(tr, attr) is None:
                t = segment_crossing(p, q, *seg)
                if t is not None:
                    setattr(tr, attr, prev[0] + t * dt)
        if tr.speed_mph is None and tr.trap_a is not None and tr.trap_b is not None:
            lo, hi = sorted((tr.trap_a, tr.trap_b))
            if hi > lo:
                tr.speed_mph = self.trap_ft / (hi - lo) * MPH_PER_FPS
                tr.n_trap_points = sum(1 for s in tr.path if lo <= s[0] <= hi) + 1

    def _resolve_movement(self, tr: VehicleTrack) -> None:
        ins = [g for _, g, d in tr.gate_events if d == "in"]
        outs = [g for _, g, d in tr.gate_events if d == "out"]
        entry = ins[0] if ins else None
        exit_ = outs[-1] if outs else None
        inferred = False
        if tr.count_pts is not None and self.count_leg:
            # The count line sits on one leg, so a vehicle that crossed it used that leg.
            if entry and not exit_ and entry != self.count_leg:
                exit_, inferred = self.count_leg, True
            elif exit_ and not entry and exit_ != self.count_leg:
                entry, inferred = self.count_leg, True
        if tr.count_dir and (entry is None) != (exit_ is None):
            # The count line also gives the direction of travel: westbound runs from the east
            # leg to the west leg. If one end was seen and fits that, fill in the other (e.g. a
            # near-lane car first seen once it's out from behind the plants).
            to = tr.count_dir.removesuffix("bound")
            frm = OPPOSITE_LEG.get(to)
            if frm in self.gates and to in self.gates:
                if exit_ == to and entry is None:
                    entry, inferred = frm, True
                elif entry == frm and exit_ is None:
                    exit_, inferred = to, True
        tr.entry, tr.exit, tr.inferred = entry, exit_, inferred
        tr.movement = classify_movement(entry, exit_) if entry and exit_ and entry != exit_ else "unknown"

    def _want_snapshot(self) -> bool:
        return any(self.snapshots_saved[m] < self.snapshot_quota
                   for m in ("through", "left_turn", "right_turn", "unknown"))

    def speed_status(self, tr: VehicleTrack) -> tuple[bool, str]:
        s = self.cfg["speed"]
        if tr.speed_mph is None:
            return False, "not_through_trap"
        if tr.movement in ("left_turn", "right_turn", "u_turn"):
            return False, "turning"
        if tr.movement != "through":
            return False, "movement_unknown"
        if tr.n_trap_points < int(s["min_points"]):
            return False, "too_few_points"
        if not (float(s["min_mph"]) <= tr.speed_mph <= float(s["max_mph"])):
            return False, "implausible_speed"
        return True, ""

    def direction_of(self, tr: VehicleTrack) -> str:
        if tr.count_dir:
            return tr.count_dir
        if tr.exit in COMPASS:
            return f"{tr.exit}bound"
        if tr.entry in COMPASS:
            opposite = {"east": "west", "west": "east", "north": "south", "south": "north"}
            return f"{opposite[tr.entry]}bound"
        return ""

    # ---------------------------------------------------------------- pedestrians
    def update_people(self, pts: float, frame, boxes, ids, vehicle_boxes, rider_boxes) -> None:
        p = self.cfg["pedestrians"]
        max_contain = float(p["max_containment_in_vehicle"])
        for box, tid in zip(boxes, ids):
            if any(containment(box, vb) >= max_contain for vb in vehicle_boxes):
                continue  # a driver or passenger seen through a window
            foot = _ground_point(box)
            tid = self.ped_alias.get(int(tid), int(tid))
            tr = self.people.get(tid)
            if tr is None:
                tr = self._rejoin(pts, foot)
                if tr is not None:
                    self.ped_alias[tid] = tr.tid
                else:
                    tr = self.people[tid] = PersonTrack(tid, first_pts=pts)
            if tr.path and tr.path[-1][0] == pts:
                continue  # already updated this frame (under its other id)
            tr.last_pts = pts
            tr.samples += 1
            if any(iou(box, rb) > 0.1 for rb in rider_boxes):
                tr.rider_samples += 1
            tr.path.append((pts, foot[0], foot[1]))
            tr.boxes.append((pts, tuple(float(v) for v in box)))
            signed = cv2.pointPolygonTest(self.crosswalk.astype(np.float32),
                                          (float(foot[0]), float(foot[1])), True)
            tr.closest_px = max(tr.closest_px, signed)
            if tr.snapshot is None and signed >= -PED_NEAR_PX and self.ped_snapshots_saved["near"] < 200:
                tr.snapshot, tr.snapshot_box = frame.copy(), tuple(float(v) for v in box)
            inside = signed >= 0
            proj = float(np.dot(np.asarray(foot) - self.cw_origin, self.cw_axis))
            if inside:
                if tr.prev_inside_pts is not None:
                    tr.inside_seconds += pts - tr.prev_inside_pts
                if tr.first_inside_pts is None:
                    tr.first_inside_pts, tr.first_proj = pts, proj
                tr.min_proj, tr.max_proj = min(tr.min_proj, proj), max(tr.max_proj, proj)
                tr.last_proj = proj
                tr.prev_inside_pts = pts
            else:
                tr.prev_inside_pts = None
            if (tr.crossed_pts is None and tr.first_inside_pts is not None
                    and tr.inside_seconds >= float(p["min_seconds_in_crosswalk"])
                    and (tr.max_proj - tr.min_proj) >= float(p["min_crossing_fraction"]) * self.cw_axis_len
                    and tr.rider_samples <= tr.samples / 2):
                tr.crossed_pts = tr.first_inside_pts
                key = "near_to_far" if tr.last_proj >= tr.first_proj else "far_to_near"
                tr.direction = self.cw_labels.get(key, key)
                self.counts["ped_total"] += 1
                self.counts[f"ped_{tr.direction}"] += 1

    def _rejoin(self, pts: float, foot) -> PersonTrack | None:
        """The person a new track at the crosswalk continues, if any: someone last seen at the
        crosswalk moments ago, nearby. People vanish behind the plants at the near end as they
        step in, and come back out under a new track id."""
        reach = 1.5 * float(self.cfg["yielding"]["waiting_area_px"])   # about where people wait

        def near(xy) -> bool:
            return cv2.pointPolygonTest(self.crosswalk.astype(np.float32),
                                        (float(xy[0]), float(xy[1])), True) >= -reach
        if not near(foot):
            return None
        best, best_d = None, self.cw_axis_len
        for tr in self.people.values():
            if tr.last_pts >= pts or not tr.path or not near(tr.path[-1][1:]):
                continue   # still being tracked, or last seen away from the crosswalk
            d = float(np.hypot(foot[0] - tr.path[-1][1], foot[1] - tr.path[-1][2]))
            if d < best_d:
                best, best_d = tr, d
        return best

    # ---------------------------------------------------------------- lifecycle
    def finalize_stale(self, pts: float, force: bool = False) -> None:
        for key in [k for k, t in self.vehicles.items() if force or pts - t.last_pts > self.finalize_after]:
            tr = self.vehicles.pop(key)
            if not tr.relevant:
                self.ignored_tracks += 1
                continue
            self._resolve_movement(tr)
            valid, _ = self.speed_status(tr)
            if valid and tr.kind == "vehicle":
                self.counts["speed_measured"] += 1
                if tr.speed_mph > self.limit:
                    self.counts["over_limit"] += 1
                if tr.speed_mph > self.limit + 10:
                    self.counts["over_limit_10"] += 1
            if (tr.snapshot is not None and tr.kind == "vehicle"
                    and self.snapshots_saved[tr.movement] < self.snapshot_quota
                    and self.on_finalize_snapshot):
                self.on_finalize_snapshot(tr)
                self.snapshots_saved[tr.movement] += 1
            tr.snapshot = None
            self.vehicle_records.append(tr)
        p = self.cfg["pedestrians"]
        min_s = float(p["min_seconds_in_crosswalk"])
        min_travel = float(p["min_crossing_fraction"]) * self.cw_axis_len
        for tid in [k for k, t in self.people.items() if force or pts - t.last_pts > self.finalize_after]:
            tr = self.people.pop(tid)
            if tr.crossed_pts is not None:
                self.ped_records.append(tr)
            if tr.closest_px >= -PED_NEAR_PX:
                reason = tr.reason(min_s, min_travel)
                self.ped_reviewed.append((tr, reason))
                if tr.snapshot is not None and self.on_ped_snapshot:
                    self.on_ped_snapshot(tr, reason)
                    self.ped_snapshots_saved["near"] += 1
            tr.snapshot = None
