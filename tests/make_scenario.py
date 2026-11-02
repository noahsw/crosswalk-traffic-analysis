"""Save a real moment from a session's tracks as a test scenario for tests/test_scenarios.py.

    uv run python tests/make_scenario.py out/live_2026-10-30_140052 P145 P153 \\
        --name group_shared_driver --note "Two people cross northbound together; ..."

Takes every pedestrian and vehicle track from shortly before the named pedestrians appeared
until shortly after they left, with times counted from the start of that window, and the
crosswalk they were judged against. No video, dates or place names: the repo is public.

The scenario's expected results start as whatever the code decides now. Watch the crossing
clip and correct them before committing: they're what the tests hold the code to.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from traffic.config import load_calibration  # noqa: E402

PAD_S = 15.0   # seconds of tracks kept before the first pedestrian appears and after the last leaves


def _r(x: float, nd: int = 1) -> float:
    return round(float(x), nd)


def dumps(scenario: dict) -> str:
    """JSON with one track per line, so the note and expected results stay easy to edit."""
    lines = ["{"]
    for i, (k, v) in enumerate(scenario.items()):
        end = "," if i < len(scenario) - 1 else ""
        if k in ("pedestrians", "vehicles"):
            rows = ",\n".join("    " + json.dumps(tr, separators=(",", ":")) for tr in v)
            lines.append(f'  "{k}": [\n{rows}\n  ]{end}')
        else:
            body = (json.dumps(v) if k in ("crosswalk", "image_size")
                    else json.dumps(v, indent=2).replace("\n", "\n  "))
            lines.append(f'  "{k}": {body}{end}')
    return "\n".join(lines) + "\n}\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path, help="a session folder with tracks.pkl")
    ap.add_argument("peds", nargs="+", help="pedestrian ids, e.g. P145")
    ap.add_argument("--name", required=True, help="file name in tests/scenarios/, without .json")
    ap.add_argument("--note", required=True, help="what happens, as seen in the clip")
    ap.add_argument("--calibration", type=Path, default=ROOT / "calibration.tapo.yaml",
                    help="the calibration the session used")
    args = ap.parse_args()

    data = pickle.load(open(args.session / "tracks.pkl", "rb"))
    cal = load_calibration(args.calibration)
    wanted = {int(p.lstrip("Pp")) for p in args.peds}
    peds = [p for p in data["pedestrians"] if p.tid in wanted]
    if {p.tid for p in peds} != wanted:
        sys.exit(f"Not in {args.session}/tracks.pkl: P{sorted(wanted - {p.tid for p in peds})}")
    lo = min(p.first_pts for p in peds) - PAD_S
    hi = max(p.last_pts for p in peds) + PAD_S

    def overlaps(tr) -> bool:
        return tr.path and tr.path[0][0] <= hi and tr.path[-1][0] >= lo

    scenario = {
        "note": args.note,
        "crosswalk": [[_r(x), _r(y)] for x, y in cal["crosswalk"]],
        "image_size": list(cal.get("image_size") or (data["width"], data["height"])),
        "pedestrians": [{
            "tid": p.tid, "direction": p.direction or "", "closest_px": _r(p.closest_px),
            "path": [[_r(t - lo, 3), _r(x), _r(y)] for t, x, y in p.path],
        } for p in data["pedestrians"] if overlaps(p)],
        "vehicles": [{
            "tid": v.tid, "kind": v.kind, "direction": v.count_dir or "",
            "path": [[_r(t - lo, 3), _r(x), _r(y)] for t, x, y in v.path],
            "boxes": [[_r(t - lo, 3), *(_r(c) for c in b)] for t, b in v.boxes],
        } for v in data["vehicles"] if v.relevant and overlaps(v)],
    }

    # Today's judgement, as a starting point for the expected results.
    sys.path.insert(0, str(Path(__file__).parent))
    from test_scenarios import judge, load   # noqa: E402
    crossings = judge(load(scenario))
    scenario["expect"] = {
        "crossings": {f"P{c.ped.tid}": {"wait_s": _r(c.wait_s)}
                      for c in crossings if c.ped.tid in wanted},
        "drivers": {f"V{e.vehicle.tid}": [e.outcome, e.phase]
                    for c in crossings if c.ped.tid in wanted for e in c.events},
    }
    out = Path(__file__).parent / "scenarios" / f"{args.name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(dumps(scenario))
    print(f"Wrote {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB). Expected results "
          "(check them against the clip):")
    print(json.dumps(scenario["expect"], indent=2))


if __name__ == "__main__":
    main()
