"""Find drivers who didn't yield to pedestrians, and cut a video clip of each one.

Runs automatically at the end of analyze.py. Run it by hand to redo the analysis from a
finished run's saved tracks (out/<run>/tracks.pkl) after changing the `yielding` settings
in config.yaml, without re-processing the video:

    uv run python yielding.py out/VIDEO
    uv run python yielding.py out/VIDEO --no-clips
"""

from __future__ import annotations

import argparse
import pickle
import shutil
from datetime import timedelta
from pathlib import Path

import pandas as pd

from traffic.clips import concat, render_clip
from traffic.config import load_calibration, load_config
from traffic.report import local, write_summary
from traffic.tracks import Analyzer
from traffic.yields import YieldAnalyzer

EVENT_COLUMNS = ["pedestrian", "pedestrian_direction", "started_waiting_local", "wait_at_curb_s",
                 "vehicle", "vehicle_direction", "vehicle_reaches_crosswalk_local", "video_seconds",
                 "pedestrian_was", "seconds_after_pedestrian_appeared", "approx_speed_mph",
                 "slowest_approach_mph", "outcome", "clip"]


def analyze_yields(out: Path, cfg: dict, cal: dict, make_clips: bool = True, log=print) -> list:
    data = pickle.loads((out / "tracks.pkl").read_bytes())
    start_time = data["start_time"]
    directions = Analyzer(cfg, cal, start_time)
    crossings = YieldAnalyzer(cfg, cal, directions.direction_of).analyze(
        data["pedestrians"], data["vehicles"])
    events = [e for c in crossings for e in c.events]
    y = cfg["yielding"]

    clip_dir = out / "yield_clips"
    shutil.rmtree(clip_dir, ignore_errors=True)
    if make_clips:
        misses = sorted((e for e in events if e.outcome == "did_not_yield"), key=lambda e: e.passes)
        misses = misses[:int(y["max_clips"])]
        if misses:
            clip_dir.mkdir()
            log(f"Rendering {len(misses)} clip(s) of drivers not yielding...")
        for i, e in enumerate(misses, 1):
            ts = start_time + timedelta(seconds=e.passes)
            path = clip_dir / f"{ts:%H%M%S}_{e.direction}_V{e.vehicle.tid}.mp4"
            render_clip(Path(data["video"]), path, e, cal["crosswalk"], start_time,
                        float(y["clip_seconds_before"]), float(y["clip_seconds_after"]))
            e.clip = f"yield_clips/{path.name}"
            log(f"  {i}/{len(misses)} {path.name}")
        if misses:
            concat([out / e.clip for e in misses], clip_dir / "all_did_not_yield.mp4")

    rows = [{
        "pedestrian": f"P{e.crossing.ped.tid}",
        "pedestrian_direction": e.crossing.direction,
        "started_waiting_local": local(start_time, e.crossing.wait_start),
        "wait_at_curb_s": round(e.crossing.wait_s, 1),
        "vehicle": f"V{e.vehicle.tid}",
        "vehicle_direction": e.direction,
        "vehicle_reaches_crosswalk_local": local(start_time, e.passes),
        "video_seconds": round(e.passes, 1),
        "pedestrian_was": e.phase,
        "seconds_after_pedestrian_appeared": round(e.lead_s, 1),
        "approx_speed_mph": round(e.speed_mph, 1) if e.speed_mph is not None else None,
        "slowest_approach_mph": round(e.min_speed_mph, 1) if e.min_speed_mph is not None else None,
        "outcome": e.outcome,
        "clip": e.clip,
    } for e in sorted(events, key=lambda e: e.passes)]
    pd.DataFrame(rows, columns=EVENT_COLUMNS).to_csv(out / "yield_events.csv", index=False)

    # Per-crossing columns on pedestrians.csv
    peds_csv = out / "pedestrians.csv"
    peds = pd.read_csv(peds_csv)
    peds = peds[[c for c in peds.columns if c not in
                 ("wait_at_curb_s", "drivers_did_not_yield", "drivers_yielded", "too_close_to_stop")]]
    per = pd.DataFrame([{
        "track_id": f"P{c.ped.tid}",
        "wait_at_curb_s": round(c.wait_s, 1),
        "drivers_did_not_yield": sum(e.outcome == "did_not_yield" for e in c.events),
        "drivers_yielded": sum(e.outcome == "yielded" for e in c.events),
        "too_close_to_stop": sum(e.outcome == "too_close_to_stop" for e in c.events),
    } for c in crossings], columns=["track_id", "wait_at_curb_s", "drivers_did_not_yield",
                                    "drivers_yielded", "too_close_to_stop"])
    peds.merge(per, on="track_id", how="left").to_csv(peds_csv, index=False)
    return events


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path, help="folder written by analyze.py")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--calibration", type=Path, default=Path("calibration.yaml"))
    ap.add_argument("--no-clips", action="store_true", help="skip rendering video clips")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cal = load_calibration(args.calibration)
    events = analyze_yields(args.out_dir, cfg, cal, make_clips=not args.no_clips)
    n = pd.Series([e.outcome for e in events]).value_counts().to_dict() if events else {}
    print(f"Yield events: {n or 'none'}")
    write_summary(args.out_dir, cfg)
    print(f"Updated {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
