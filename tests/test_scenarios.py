"""Real moments at the crosswalk, replayed from their tracks (see make_scenario.py).

Each scenario's expected results were checked against its crossing clip. Both ways of judging
must get them: all at once, as yielding.py does, and as live mode does, crossing by crossing
while the tracks come in.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from traffic.config import load_config
from traffic.live import LiveYieldWatcher
from traffic.tracks import PersonTrack, VehicleTrack
from traffic.yields import YieldAnalyzer

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config.yaml")
SCENARIOS = sorted((Path(__file__).parent / "scenarios").glob("*.json"))
FINALIZE_S = 3.0          # tracks.finalize_after_seconds: how long after last seen a track is done
WAIT_TOLERANCE_S = 1.5


def load(scenario: dict) -> SimpleNamespace:
    peds = []
    for p in scenario["pedestrians"]:
        tr = PersonTrack(p["tid"], first_pts=p["path"][0][0], direction=p["direction"])
        tr.path = [tuple(x) for x in p["path"]]
        tr.last_pts, tr.closest_px = tr.path[-1][0], p["closest_px"]
        peds.append(tr)
    vehicles = []
    for v in scenario["vehicles"]:
        tr = VehicleTrack(v["tid"], v["kind"], 5, count_dir=v["direction"] or None)
        tr.path = [tuple(x) for x in v["path"]]
        tr.boxes = [(b[0], tuple(b[1:])) for b in v["boxes"]]
        tr.first_pts, tr.last_pts = tr.path[0][0], tr.path[-1][0]
        tr.count_pts = tr.first_pts   # only tracks that mattered were saved
        vehicles.append(tr)
    cal = {"crosswalk": scenario["crosswalk"], "image_size": scenario["image_size"]}
    return SimpleNamespace(cal=cal, peds=peds, vehicles=vehicles, expect=scenario.get("expect"))


def analyzer(s: SimpleNamespace) -> YieldAnalyzer:
    return YieldAnalyzer(CFG, s.cal, lambda v: v.count_dir or "")


def judge(s: SimpleNamespace) -> list:
    return analyzer(s).analyze(s.peds, s.vehicles)


def judge_live(s: SimpleNamespace) -> list:
    """Every driver's event as live mode judges them: tracks finish FINALIZE_S after they were
    last seen, and the watcher is polled every second."""
    a = SimpleNamespace(direction_of=lambda v: v.count_dir or "", ped_records=[], vehicle_records=[],
                        people={}, vehicles={})
    watcher = LiveYieldWatcher(CFG, s.cal, a)
    watcher.ya = analyzer(s)
    end = max(tr.last_pts for tr in s.peds + s.vehicles) + FINALIZE_S
    events = []
    now = 0.0
    while now <= end + LiveYieldWatcher.SETTLE_SECONDS:
        done = lambda tr: tr.last_pts + FINALIZE_S <= now   # noqa: E731
        seen = lambda tr: tr.first_pts <= now               # noqa: E731
        a.ped_records = sorted((p for p in s.peds if done(p)), key=lambda p: p.last_pts)
        a.people = {p.tid: p for p in s.peds if seen(p) and not done(p)}
        a.vehicle_records = sorted((v for v in s.vehicles if done(v)), key=lambda v: v.last_pts)
        a.vehicles = {v.tid: v for v in s.vehicles if seen(v) and not done(v)}
        events += [e for j in watcher.poll(now) for e in j.events]
        now += 1.0
    events += [e for j in watcher.poll(now, force=True) for e in j.events]
    return events


def _drivers(events) -> dict:
    return {f"V{e.vehicle.tid}": [e.outcome, e.phase] for e in events}


@pytest.mark.parametrize("path", SCENARIOS, ids=[p.stem for p in SCENARIOS])
def test_scenario(path):
    s = load(json.loads(path.read_text()))
    crossings = {f"P{c.ped.tid}": c for c in judge(s)}
    for name, want in s.expect["crossings"].items():
        c = crossings.get(name)
        assert c is not None, f"{name} wasn't seen crossing"
        assert abs(c.wait_s - want["wait_s"]) <= WAIT_TOLERANCE_S, \
            f"{name} waited {c.wait_s:.1f} s, expected about {want['wait_s']} s"
    theirs = {c.ped.tid for c in crossings.values()} & {int(n[1:]) for n in s.expect["crossings"]}
    batch = _drivers(e for c in crossings.values() if c.ped.tid in theirs for e in c.events)
    assert batch == s.expect["drivers"]
    live = {k: v for k, v in _drivers(judge_live(s)).items() if k in batch or k in s.expect["drivers"]}
    assert live == s.expect["drivers"], "live mode judged them differently"
