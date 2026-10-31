"""Yield classification on synthetic tracks laid over the real crosswalk geometry."""

from pathlib import Path

import cv2
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


def ped_crossing_near_to_far(ya: YieldAnalyzer, t0: float, seconds: float = 8.0) -> PersonTrack:
    """Waits 4 s at the near (north) curb, then walks to the far curb."""
    p = PersonTrack(2, first_pts=t0, direction="southbound")
    near = ya.origin - ya.axis * 30
    far = ya.origin + ya.axis * (ya.length_px + 30)
    for t in np.arange(t0, t0 + 4.0, 0.1):
        p.path.append((t, *near))
    for k, t in enumerate(np.arange(t0 + 4.0, t0 + 4.0 + seconds, 0.1)):
        f = k / (seconds * 10)
        p.path.append((t, *(near + (far - near) * f)))
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
    assert 102.5 < c.enter < 103.0 and c.exit > c.enter + 5   # starts 30 px behind the curb


def test_classifies_each_driver():
    ya = analyzer()
    ped = ped_crossing_far_to_near(ya, 100.0)    # waits 100-102, crosses 102-110
    vs = [
        vehicle(ya, 1, "eastbound", 0.75, passes=103.5, mph=22),            # far lane, ped in its half
        vehicle(ya, 2, "westbound", 0.25, passes=112.0, mph=15, stop_for=6),  # stops, goes after
        vehicle(ya, 3, "eastbound", 0.75, passes=100.8, mph=25),            # 0.8 s after ped appeared
        vehicle(ya, 4, "westbound", 0.25, passes=102.5, mph=25),            # ped still at far curb
    ]
    got = outcomes(ya.analyze([ped], vs))
    assert got[1] == ("did_not_yield", "in_crosswalk")
    assert got[2][0] == "yielded"
    assert got[3][0] == "too_close_to_stop"
    # A pedestrian waiting at the far curb needs the near (westbound) lane to stop too.
    assert got[4] == ("too_close_to_stop", "waiting_at_curb")


def test_waiting_counts_only_the_adjacent_lane_when_set():
    ya = analyzer()
    ya.waiting_both_lanes = False
    ped = ped_crossing_far_to_near(ya, 100.0)    # waits at the far curb 100-102
    got = outcomes(ya.analyze([ped], [vehicle(ya, 4, "westbound", 0.25, passes=102.5, mph=25)]))
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


def test_stop_before_pedestrian_needed_the_lane_is_not_a_yield():
    ya = analyzer()
    ped = ped_crossing_far_to_near(ya, 100.0)    # waits at the far curb 100-102, steps in ~102.8
    # Westbound (near lane): stops 100.5-102.2 while the pedestrian waits across the street (a
    # stop sign, a turn from Havana), then drives through at 103.2, after they've stepped in.
    v = VehicleTrack(7, "vehicle", 5, count_dir="westbound")
    pps = 15 / MPH_PER_FPS * ya.px_per_ft
    cross_pt = ya.origin + ya.axis * ya.length_px * 0.25
    for t in np.arange(98.0, 106.0, 0.0667):
        d = -pps * (1.0 + max(0.0, 100.5 - t)) if t < 102.2 else -pps * (103.2 - t)
        v.path.append((t, *(cross_pt + ROAD * d)))
    v.boxes = [(t, (x - 80, y - 60, x + 80, y)) for t, x, y in v.path]
    got = outcomes(ya.analyze([ped], [v]))
    assert got[7] == ("did_not_yield", "in_crosswalk")


def test_waiting_through_the_crossing_is_a_yield():
    ya = analyzer()
    ped = ped_crossing_far_to_near(ya, 100.0)
    # Westbound: stops at 99, before the pedestrian appears, and waits until they're across.
    got = outcomes(ya.analyze([ped], [vehicle(ya, 8, "westbound", 0.25, passes=111.0, mph=15,
                                              stop_for=12)]))
    assert got[8][0] == "yielded"


def test_stall_after_being_hidden_is_not_a_stop():
    ya = analyzer()
    ped = ped_crossing_far_to_near(ya, 100.0)
    # Westbound at 15 mph, hidden 104.0-104.6 (behind a pole), passing at 105 with the pedestrian
    # in the crosswalk. Just after the gap the smoothed position lags, which looks like slowing.
    v = vehicle(ya, 9, "westbound", 0.25, passes=105.0, mph=15)
    hidden = [k for k, (t, _, _) in enumerate(v.path) if 104.0 <= t < 104.6]
    stall = v.path[hidden[0] - 1][1:]
    after = range(hidden[-1] + 1, hidden[-1] + 3)
    v.path = [(t, *stall) if k in after else (t, x, y)
              for k, (t, x, y) in enumerate(v.path) if k not in hidden]
    v.boxes = [(t, (x - 80, y - 60, x + 80, y)) for t, x, y in v.path]
    got = outcomes(ya.analyze([ped], [v]))
    assert got[9] == ("did_not_yield", "in_crosswalk")


def test_stopping_then_pulling_out_in_front_is_not_a_yield():
    ya = analyzer()
    ped = ped_crossing_near_to_far(ya, 100.0)    # waits at the near curb 100-104, then crosses
    # Westbound (near lane): stops 101.5-102.5 (a stop sign), then drives past at 103.5 while
    # the pedestrian is still waiting beside its lane.
    v = VehicleTrack(10, "vehicle", 5, count_dir="westbound")
    pps = 15 / MPH_PER_FPS * ya.px_per_ft
    cross_pt = ya.origin + ya.axis * ya.length_px * 0.25
    for t in np.arange(97.0, 106.0, 0.0667):
        d = -pps * (1.0 + max(0.0, 101.5 - t)) if t < 102.5 else -pps * (103.5 - t)
        v.path.append((t, *(cross_pt + ROAD * d)))
    v.boxes = [(t, (x - 80, y - 60, x + 80, y)) for t, x, y in v.path]
    got = outcomes(ya.analyze([ped], [v]))
    assert got[10] == ("did_not_yield", "waiting_at_curb")


def test_car_hiding_their_feet_does_not_end_the_wait():
    ya = analyzer()
    ped = ped_crossing_far_to_near(ya, 100.0)    # waits at the far curb 100-102
    # At 101, a car passing in front hides their legs: the box gets shorter, so their feet seem
    # to step back from the curb for a moment.
    ped.path = [(t, x, y - 40) if 101.0 <= t < 101.3 else (t, x, y) for t, x, y in ped.path]
    c = ya.crossing_for(ped)
    assert abs(c.wait_start - 100.0) < 0.3


def test_one_clip_per_pedestrian():
    from datetime import datetime

    from traffic.clips import by_pedestrian, clip_name, clip_window

    ya = analyzer()
    first = ped_crossing_far_to_near(ya, 100.0)     # waits 100-102, crosses 102-110
    later = ped_crossing_near_to_far(ya, 200.0)
    later.tid = 3
    vs = [
        vehicle(ya, 1, "eastbound", 0.75, passes=103.5, mph=22),
        vehicle(ya, 2, "westbound", 0.25, passes=105.0, mph=20),
        vehicle(ya, 5, "westbound", 0.25, passes=205.0, mph=20),
    ]
    events = [e for c in ya.analyze([first, later], vs) for e in c.events
              if e.outcome == "did_not_yield"]
    groups = by_pedestrian(events)
    assert [[e.vehicle.tid for e in g] for g in groups] == [[1, 2], [5]]
    assert clip_name(datetime(2026, 10, 30, 14, 0), groups[0]) == \
        "20261030_140143_did_not_yield_eastbound+westbound_V1+V2.mp4"
    t0, t1 = clip_window(groups[0], 6.0, 3.0)
    assert t0 <= 103.5 - 4 and t1 == groups[0][0].crossing.exit + 3 > 105.0 + 3   # once they're across


def test_clip_runs_until_the_pedestrian_is_across():
    from traffic.clips import clip_window
    ya = analyzer()
    ped = ped_crossing_near_to_far(ya, 100.0)    # waits 100-104, crosses 104-112
    v = vehicle(ya, 1, "westbound", 0.25, passes=103.5, mph=25)   # passes while they wait
    (c,) = ya.analyze([ped], [v])
    (e,) = c.events
    assert (e.outcome, e.phase) == ("did_not_yield", "waiting_at_curb")
    t0, t1 = clip_window([e], 6.0, 3.0)
    assert t0 < e.passes and t1 == c.exit + 3.0 > e.passes + 3.0


def test_live_judges_a_driver_against_whoever_was_waiting_first():
    from types import SimpleNamespace

    from traffic.live import LiveYieldWatcher

    ya = analyzer()
    first = ped_crossing_near_to_far(ya, 100.0)     # waits 100-104, crosses 104-112...
    for t in np.arange(112.0, 125.0, 0.1):          # ...then stays in view on the far side
        first.path.append((t, *first.path[-1][1:]))
    first.last_pts = first.path[-1][0]
    second = ped_crossing_near_to_far(ya, 103.0)    # steps up as the car arrives, leaves first
    second.tid = 3
    v = vehicle(ya, 1, "westbound", 0.25, passes=103.7, mph=25)
    a = SimpleNamespace(direction_of=lambda v: v.count_dir, ped_records=[second, first],
                        vehicles={}, vehicle_records=[v], people={})
    watcher = LiveYieldWatcher(CFG, CAL, a)
    assert watcher.poll(131.0) == []                # second has settled; first hasn't
    judged = watcher.poll(141.0)
    events = [e for j in judged for e in j.events]
    assert [(e.vehicle.tid, e.outcome, e.crossing.ped.tid) for e in events] == \
        [(1, "did_not_yield", first.tid)]


# Waiting means standing still near the curb, not just passing by (see crossing_for).

def far_curb(ya: YieldAnalyzer, along_px: float = 0.0, into_road_ft: float = 0.0) -> np.ndarray:
    """A spot at the far curb: `along_px` along it from the middle of the crosswalk's end (positive
    toward the second corner), `into_road_ft` out from it into the road."""
    cw = ya.crosswalk.astype(float)
    along = (cw[2] - cw[3]) / np.linalg.norm(cw[2] - cw[3])
    return (cw[2] + cw[3]) / 2 + along * along_px - ya.axis * into_road_ft * ya.px_per_ft


def ped_walking(ya: YieldAnalyzer, legs, t0: float = 100.0, tid: int = 1) -> PersonTrack:
    """A pedestrian following `legs`, (seconds, from, to) each, at 10 samples a second (standing
    still when from and to are the same), then crossing from the far curb to the near one in 8 s."""
    p = PersonTrack(tid, first_pts=t0, direction="northbound")
    legs = list(legs) + [(8.0, far_curb(ya, 0, 1), ya.origin - ya.axis * 30)]
    t = t0
    for seconds, a, b in legs:
        n = int(round(seconds * 10))
        for k in range(n):
            p.path.append((round(t, 3), *(a + (b - a) * k / n)))
            t += 0.1
    p.boxes = [(t, (x - 10, y - 60, x + 10, y)) for t, x, y in p.path]
    p.last_pts = p.path[-1][0]
    return p


def width(ya: YieldAnalyzer) -> float:
    cw = ya.crosswalk
    return float(np.linalg.norm(cw[2] - cw[3]))


def test_standing_still_at_the_curbs_edge_beside_the_crosswalk_is_waiting():
    # A real miss: standing just off the curb a few feet beside the crosswalk, a little past
    # the waiting area, while two cars drove by.
    ya = analyzer()
    spot = far_curb(ya, width(ya), into_road_ft=1)
    assert cv2.pointPolygonTest(ya.wait_far, tuple(map(float, spot)), False) < 0
    ped = ped_walking(ya, [(8.0, spot, spot), (2.0, spot, far_curb(ya, 0, 1))])
    car = vehicle(ya, 7, "eastbound", 0.75, passes=104.0, mph=25)
    (c,) = ya.analyze([ped], [car])
    assert abs(c.wait_start - 100.0) < 0.3
    assert outcomes([c]) == {7: ("did_not_yield", "waiting_at_curb")}


def test_standing_still_in_the_gutter_is_waiting():
    ya = analyzer()
    spot = far_curb(ya, width(ya), into_road_ft=4)
    ped = ped_walking(ya, [(8.0, spot, spot), (2.0, spot, far_curb(ya, 0, 1))])
    assert abs(ya.crossing_for(ped).wait_start - 100.0) < 0.3


def test_standing_still_well_out_in_the_road_is_not_waiting():
    # Further out than the waiting areas reach: maybe by a parked car, not at the curb.
    ya = analyzer()
    spot = far_curb(ya, width(ya), into_road_ft=9)
    ped = ped_walking(ya, [(8.0, spot, spot), (2.0, spot, far_curb(ya, 0, 1))])
    car = vehicle(ya, 7, "eastbound", 0.75, passes=104.0, mph=25)
    (c,) = ya.analyze([ped], [car])
    assert c.wait_start > 108.0
    assert outcomes([c]) == {}


def test_standing_still_far_along_the_curb_is_not_waiting():
    # Beyond the waiting area along the sidewalk (waiting for a ride, say), then walking over.
    ya = analyzer()
    spot = far_curb(ya, 2.5 * width(ya), into_road_ft=1)
    ped = ped_walking(ya, [(8.0, spot, spot), (4.0, spot, far_curb(ya, 0, 1))])
    car = vehicle(ya, 7, "eastbound", 0.75, passes=104.0, mph=25)
    (c,) = ya.analyze([ped], [car])
    assert c.wait_start > 108.0
    assert outcomes([c]) == {}


def test_walking_along_the_gutter_to_the_crosswalk_is_not_waiting():
    # Someone walking along the edge of the road at 4 ft/s who turns into the crosswalk hasn't
    # been waiting until they reach its end: a car passing them on the way doesn't count.
    ya = analyzer()
    start = far_curb(ya, 2.5 * width(ya), into_road_ft=1)
    walk = 2.5 * width(ya) / (4 * ya.px_per_ft)
    ped = ped_walking(ya, [(walk, start, far_curb(ya, 0, 1))])
    car = vehicle(ya, 7, "eastbound", 0.75, passes=100.0 + walk - 4.0, mph=25)   # passes them
    (c,) = ya.analyze([ped], [car])
    assert c.wait_s < 2.5     # from reaching the crosswalk's end (half its width + 10 px away)
    assert outcomes([c]) == {}


def test_walking_to_the_crosswalk_after_waiting_beside_it_keeps_the_wait():
    # Waiting beside the crosswalk, then walking slowly (2 ft/s, with a child) along the gutter
    # to it: more than 3 s away from the waiting area, but the wait goes on.
    ya = analyzer()
    spot = far_curb(ya, width(ya), into_road_ft=1)
    walk = width(ya) / (2 * ya.px_per_ft)
    assert walk > 3.0
    ped = ped_walking(ya, [(8.0, spot, spot), (walk, spot, far_curb(ya, 0, 1))])
    car = vehicle(ya, 7, "eastbound", 0.75, passes=104.0, mph=25)
    (c,) = ya.analyze([ped], [car])
    assert abs(c.wait_start - 100.0) < 0.3
    assert outcomes([c]) == {7: ("did_not_yield", "waiting_at_curb")}


def test_shifting_about_while_standing_is_still_standing():
    # Detection boxes jitter by a few pixels from frame to frame, and people shift their weight.
    ya = analyzer()
    spot = far_curb(ya, width(ya), into_road_ft=1)
    ped = ped_walking(ya, [(8.0, spot, spot), (2.0, spot, far_curb(ya, 0, 1))])
    rng = np.random.default_rng(0)
    ped.path = [(t, x + rng.uniform(-4, 4), y + rng.uniform(-4, 4)) if t < 108 else (t, x, y)
                for t, x, y in ped.path]
    assert abs(ya.crossing_for(ped).wait_start - 100.0) < 0.3


def test_standing_by_the_crosswalk_without_crossing_is_not_a_crossing():
    ya = analyzer()
    spot = far_curb(ya, width(ya), into_road_ft=1)
    p = PersonTrack(1, first_pts=100.0)
    p.path = [(100.0 + k / 10, *spot) for k in range(300)]
    p.path += [(130.0 + k / 10, *(spot + (far_curb(ya, 3 * width(ya)) - spot) * k / 50)) for k in range(50)]
    assert ya.crossing_for(p) is None


# Cars seen only just before the crosswalk: coming into view at the frame's edge, or picked up
# again under a new id after the tracker lost them.

def seen_from(v: VehicleTrack, t0: float) -> VehicleTrack:
    """The vehicle as tracked only from t0 on."""
    v.path = [p for p in v.path if p[0] >= t0]
    v.boxes = [b for b in v.boxes if b[0] >= t0]
    v.first_pts = v.path[0][0]
    return v


def test_car_seen_just_before_the_crosswalk_driving_through_did_not_yield():
    ya = analyzer()
    ped = ped_crossing_near_to_far(ya, 100.0)    # waits 100-104, crosses 104-112
    v = seen_from(vehicle(ya, 7, "westbound", 0.25, passes=103.5, mph=25), 103.2)
    assert not ya.speed_mph(v, 102.5, 103.5)     # too little track to measure its approach
    got = outcomes(ya.analyze([ped], [v]))
    assert got == {7: ("did_not_yield", "waiting_at_curb")}


def test_car_seen_just_before_the_crosswalk_creeping_is_not_judged():
    # Barely moving at the crosswalk: it may have stopped first, and there's no telling.
    ya = analyzer()
    ped = ped_crossing_near_to_far(ya, 100.0)
    v = seen_from(vehicle(ya, 7, "westbound", 0.25, passes=103.5, mph=3), 103.2)
    assert outcomes(ya.analyze([ped], [v])) == {}


def test_car_seen_just_before_the_crosswalk_after_they_cleared_its_lane_is_not_judged():
    ya = analyzer()
    ped = ped_crossing_near_to_far(ya, 100.0)    # past the near (westbound) half by about 109
    v = seen_from(vehicle(ya, 7, "westbound", 0.25, passes=110.5, mph=25), 110.2)
    assert outcomes(ya.analyze([ped], [v])) == {}
