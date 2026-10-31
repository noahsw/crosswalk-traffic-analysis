"""Saved crossings replayed from their video: detection, tracking and judging, end to end.

Slow, and only on this computer: each scenario in tests/scenarios/ that has its crossing's raw
video at tests/replay_clips/<scenario name>.mp4 is played through live.py in real time, and the
drivers it judges are compared with the scenario's expected results. Scenarios without a clip
are skipped. Run with

    PYTHONPATH=. uv run pytest -m replay

The raw videos come from live mode's raw_crossings/ folder (see the README). They show real
people, so tests/replay_clips/ is gitignored and must stay that way: the repo is public.

Track ids aren't the same in a replay, so each expected pedestrian is matched to a replayed
crossing in the same direction (the closest wait, for people crossing together), and their
drivers are compared by direction, outcome and phase.
"""

import csv
import json
import subprocess
import sys
from collections import Counter
from itertools import permutations
from pathlib import Path

import pytest

from traffic.config import load_calibration, load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config.yaml")
CLIPS = Path(__file__).parent / "replay_clips"
SCENARIOS = sorted((Path(__file__).parent / "scenarios").glob("*.json"))
WAIT_TOLERANCE_S = 2.0
PHASES = {"waiting to cross": "waiting_at_curb", "in the crosswalk": "in_crosswalk"}

pytestmark = pytest.mark.replay


def expected(scenario: dict) -> tuple[list, Counter]:
    """([(direction, wait_s)] for each expected crossing, and a Counter of (direction, outcome,
    phase) for every expected driver)."""
    peds = {f"P{p['tid']}": p for p in scenario["pedestrians"]}
    vehicles = {f"V{v['tid']}": v for v in scenario["vehicles"]}
    drivers = Counter((vehicles[v]["direction"], outcome, phase)
                      for v, (outcome, phase) in scenario["expect"]["drivers"].items())
    return [(peds[p]["direction"], want["wait_s"])
            for p, want in scenario["expect"]["crossings"].items()], drivers


def calibration_for(scenario: dict) -> Path | None:
    """The calibration the scenario's crosswalk came from: live.calibration (the camera's), or
    another calibration*.yaml for a scenario from a recording."""
    def crosswalk(path):
        return [[round(float(x), 1), round(float(y), 1)] for x, y in load_calibration(path)["crosswalk"]]
    paths = [ROOT / CFG["live"]["calibration"], *sorted(ROOT.glob("calibration*.yaml"))]
    return next((p for p in paths if p.exists() and crosswalk(p) == scenario["crosswalk"]), None)


def replay(clip: Path, calibration: Path, out: Path) -> None:
    cmd = [sys.executable, "live.py", str(clip), "--calibration", str(calibration), "--out", str(out),
           "--work", str(out.parent / "work"), "--no-clips", "--progress-every", "3600"]
    run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    assert run.returncode == 0, f"live.py failed:\n{run.stdout[-3000:]}\n{run.stderr[-3000:]}"


def _rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def judged_batch(out: Path) -> list[tuple[str, float, dict]]:
    """(direction, wait_s, {vehicle: (direction, outcome, phase)} of its drivers) for each
    crossing, in order, as yielding.py judged the replay's tracks."""
    events = _rows(out / "yield_events.csv")
    peds = sorted(_rows(out / "pedestrians.csv"), key=lambda p: float(p["video_seconds"]))
    return [(p["direction"], float(p["wait_at_curb_s"] or 0),
             {e["vehicle"]: (e["vehicle_direction"], e["outcome"], e["pedestrian_was"]) for e in events
              if p["track_id"] in e["pedestrian"].split("+")})
            for p in peds if p["wait_at_curb_s"] != ""]


def judged_live(out: Path) -> list[tuple[str, None, dict]]:
    """The same, as live mode judged them while the replay ran (yield_log.csv lists each crossing
    followed by the drivers judged on it; it doesn't give the wait)."""
    crossings = []
    for r in _rows(out / "yield_log.csv"):
        if r["outcome"] == "pedestrian_crossing":
            crossings.append((r["time_local"], r["direction"], None, {}))
        elif crossings:
            phase = next((p for text, p in PHASES.items() if text in r["what"]), "")
            crossings[-1][3][f"V{r['vehicle_id']}"] = (r["direction"], r["outcome"], phase)
    return [c[1:] for c in sorted(crossings, key=lambda c: c[0])]


def match(want: list, got: list) -> tuple[list, Counter]:
    """Each expected crossing's replayed one (None if there are too few in its direction), and a
    Counter of their drivers (each once, however many of them it was judged against). In each direction, the replayed crossings whose waits are closest
    to the expected ones (or the first ones, with no waits to go by)."""
    found = [None] * len(want)
    for direction in {w[0] for w in want}:
        mine = [i for i, w in enumerate(want) if w[0] == direction]
        theirs = [g for g in got if g[0] == direction]
        if len(theirs) < len(mine):
            continue
        best = min(permutations(theirs, len(mine)), key=lambda gs: sum(
            abs(g[1] - want[i][1]) for i, g in zip(mine, gs) if g[1] is not None))
        for i, g in zip(mine, best):
            found[i] = g
    drivers = Counter({v: d for g in found if g for v, d in g[2].items()}.values())
    return found, drivers


@pytest.mark.parametrize("path", SCENARIOS, ids=[p.stem for p in SCENARIOS])
def test_replay(path, tmp_path):
    clip = CLIPS / f"{path.stem}.mp4"
    if not clip.exists():
        pytest.skip(f"no raw clip at {clip.relative_to(ROOT)}")
    scenario = json.loads(path.read_text())
    calibration = calibration_for(scenario)
    if calibration is None:
        pytest.skip(f"no calibration*.yaml has the crosswalk {path.name} was judged against")

    replay(clip, calibration, tmp_path / "out")
    want_crossings, want_drivers = expected(scenario)
    for how, got in (("batch", judged_batch(tmp_path / "out")), ("live", judged_live(tmp_path / "out"))):
        found, drivers = match(want_crossings, got)
        for (direction, wait_s), g in zip(want_crossings, found):
            assert g is not None, f"{how}: no {direction} crossing seen; saw {[g[0] for g in got]}"
            if g[1] is not None:
                assert abs(g[1] - wait_s) <= WAIT_TOLERANCE_S, \
                    f"{how}: the {direction} pedestrian waited {g[1]:.1f} s, expected about {wait_s} s"
        assert drivers == want_drivers, f"{how}: judged drivers differ (direction, outcome, phase)"
