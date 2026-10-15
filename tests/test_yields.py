"""Yield classification on synthetic tracks laid over the real crosswalk geometry."""

from pathlib import Path

import numpy as np

from traffic.config import load_calibration, load_config
from traffic.tracks import MPH_PER_FPS, PersonTrack, VehicleTrack
from traffic.yields import YieldAnalyzer

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config.yaml")
CAL = load_calibration(ROOT / "calibration.example.yaml")
ROAD = np.array([0.966, 0.259])   # on-screen direction of travel for westbound (left to right)


def analyzer() -> YieldAnalyzer:
    return YieldAnalyzer(CFG, CAL, lambda v: v.count_dir)


def ped_crossing_far_to_near(ya: YieldAnalyzer, t0: float, seconds: float = 8.0) -> PersonTrack:
    """Waits 2 s at the far (south) curb, then walks to the near curb."""
    p = PersonTrack(1, first_pts=t0, direction="northbound")
    far = ya.origin + ya.axis * (ya.length_px + 30)
    near = ya.origin - ya.axis * 30
    for t in np.arange(t0, t0 + 2.0, 0.1):
        p.path.append((t, *far))
    for k, t in enumerate(np.arange(t0 + 2.0, t0 + 2.0 + seconds, 0.1)):
        f = k / (seconds * 10)
        p.path.append((t, *(far + (near - far) * f)))
    p.boxes = [(t, (x - 10, y - 60, x + 10, y)) for t, x, y in p.path]
    p.last_pts = p.path[-1][0]
    return p


def vehicle(ya: YieldAnalyzer, tid: int, direction: str, lane_frac: float, passes: float,
            mph: float, stop_for: float = 0.0) -> VehicleTrack:
    """Vehicle travelling along the road through the crosswalk at `lane_frac` of its length.

    With stop_for > 0 it stops 60 px before the crosswalk for that long, then continues.
    """
    v = VehicleTrack(tid, "vehicle", 5, count_dir=direction)
    sign = 1 if direction == "westbound" else -1
    cross_pt = ya.origin + ya.axis * ya.length_px * lane_frac
    pps = mph / MPH_PER_FPS * ya.px_per_ft
    for t in np.arange(passes - 10, passes + 3, 0.0667):
        if stop_for and t < passes:
            # Drive up, stop 60 px short, wait, then pull through and reach the crosswalk at `passes`.
            stop_start = passes - 60 / pps - stop_for
            if t < stop_start:
                d = -60 - pps * (stop_start - t)
            elif t < passes - 60 / pps:
                d = -60
            else:
                d = -pps * (passes - t)
        else:
            d = pps * (t - passes)
        x, y = cross_pt + sign * ROAD * d
        v.path.append((t, x, y))
    v.boxes = [(t, (x - 80, y - 60, x + 80, y)) for t, x, y in v.path]
    v.first_pts, v.last_pts = v.path[0][0], v.path[-1][0]
    return v


def outcomes(crossings):
    return {e.vehicle.tid: (e.outcome, e.phase) for c in crossings for e in c.events}


def test_crossing_window():
    ya = analyzer()
    c = ya.crossing_for(ped_crossing_far_to_near(ya, 100.0))
    assert c is not None and c.start_end == "far"
    assert abs(c.wait_start - 100.0) < 0.3      # waiting at the curb counts
    assert 102.5 < c.enter < 103.5 and c.exit > c.enter + 5   # starts 30 px behind the curb


def test_classifies_each_driver():
    ya = analyzer()
    ped = ped_crossing_far_to_near(ya, 100.0)    # waits 100-102, crosses 102-110
    vs = [
        vehicle(ya, 1, "eastbound", 0.75, passes=103.5, mph=22),            # far lane, ped in its half
        vehicle(ya, 2, "westbound", 0.25, passes=112.0, mph=15, stop_for=6),  # stops, goes after
        vehicle(ya, 3, "eastbound", 0.75, passes=100.8, mph=25),            # 0.8 s after ped appeared
        vehicle(ya, 4, "westbound", 0.25, passes=103.0, mph=25),            # ped still at far curb
    ]
    got = outcomes(ya.analyze([ped], vs))
    assert got[1] == ("did_not_yield", "in_crosswalk")
    assert got[2][0] == "yielded"
    assert got[3][0] == "too_close_to_stop"
    # A pedestrian waiting at the far curb only needs the far (eastbound) lane to stop.
    assert 4 not in got


def test_cleared_half_is_not_a_conflict():
    ya = analyzer()
    ped = ped_crossing_far_to_near(ya, 100.0)
    # Eastbound (far lane) vehicle passes late in the crossing, after the ped left the far half.
    got = outcomes(ya.analyze([ped], [vehicle(ya, 5, "eastbound", 0.75, passes=109.5, mph=22)]))
    assert 5 not in got


def test_stop_after_pedestrian_finished_is_not_a_yield():
    ya = analyzer()
    ped = ped_crossing_far_to_near(ya, 100.0)    # done crossing by ~110
    # Westbound vehicle that stops 113-117 (for something else), long after the pedestrian left.
    got = outcomes(ya.analyze([ped], [vehicle(ya, 6, "westbound", 0.25, passes=118.0, mph=15, stop_for=4)]))
    assert 6 not in got
