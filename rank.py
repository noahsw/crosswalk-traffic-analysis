"""Rank every driver who did not yield, across all the runs and live sessions in out/, worst first.

    uv run python rank.py              # rank, and join the worst 10 clips
    uv run python rank.py --top 20
    uv run python rank.py --dry-run    # show what would be renamed, change nothing

Writes:
- out/worst_drivers.csv: one row per driver who did not yield, worst first, with its clip.
- out/worst_did_not_yield.mp4: the clips of the worst drivers (--top), joined, worst first.

Each driver is scored from 1 to 5 (see traffic/yields.py, severity). Runs since scores were added
already have them. Older ones are scored here, and their clips are renamed to start with the
score (S4_...), with yield_events.csv and yield_log.csv updated to the new names:

- A finished run is scored from yield_events.csv and its saved tracks, in full.
- A live session that never finished (no tracks.pkl: the Mac crashed, say) is scored from its
  yield_log.csv, which only has whether the pedestrian was waiting or in the crosswalk and the
  speed. Its yield_log.csv is left alone if it changed in the last 2 hours, in case the session
  is still running.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from pathlib import Path

import pandas as pd

from traffic.clips import reel
from traffic.config import load_calibration, load_config
from traffic.yields import YieldAnalyzer, severity

ROOT = Path(__file__).resolve().parent
COLUMNS = ["severity", "when", "run", "direction", "approx_speed_mph", "pedestrian_was",
           "severity_why", "scored_from", "clip"]
SCORE_PREFIX = re.compile(r"^S\d_")
RUNNING_SECONDS = 2 * 3600


def _num(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _scored(rows: pd.DataFrame) -> bool:
    return "severity" in rows.columns and bool((rows["severity"] != "").all())


def _calibration(run: Path, fallback: Path) -> dict:
    meta = run / "run_meta.json"
    name = json.loads(meta.read_text())["calibration"].get("file") if meta.exists() else None
    for path in ([Path(name), ROOT / name] if name else []) + [fallback]:
        if path.exists():
            return load_calibration(path)
    raise FileNotFoundError(f"no calibration file for {run}")


def from_events(run: Path, cfg: dict, fallback_cal: Path) -> tuple[list[dict], pd.DataFrame | None]:
    """Drivers who did not yield in a finished run, and its yield_events.csv with scores added
    (None if it already had them)."""
    ev = pd.read_csv(run / "yield_events.csv", dtype=str, keep_default_na=False)
    missed = ev["outcome"] == "did_not_yield"
    updated = None
    if missed.any() and not _scored(ev[missed]):
        # Score from the CSV, plus the pedestrian's position from the saved tracks.
        ya = YieldAnalyzer(cfg, _calibration(run, fallback_cal), direction_of=None)
        peds = {f"P{p.tid}": p for p in pickle.loads((run / "tracks.pkl").read_bytes())["pedestrians"]}
        scores, whys = [], []
        for r in ev.itertuples():
            if r.outcome != "did_not_yield":
                scores.append("")
                whys.append("")
                continue
            crossings = [c for p in r.pedestrian.split("+") if p in peds
                         for c in [ya.crossing_for(peds[p])] if c is not None]
            beside = (ya.beside(crossings, r.vehicle_direction, r.pedestrian_was, float(r.video_seconds))
                      if crossings else None)
            score, why = severity(r.pedestrian_was, beside, _num(r.approx_speed_mph),
                                  _num(r.slowest_approach_mph), _num(r.seconds_after_pedestrian_appeared))
            scores.append(str(score))
            whys.append(why)
        ev = ev.drop(columns=["severity", "severity_why"], errors="ignore")
        at = ev.columns.get_loc("clip")
        ev.insert(at, "severity_why", whys)
        ev.insert(at, "severity", scores)
        updated = ev
    rows = [{"severity": int(r.severity), "when": r.vehicle_reaches_crosswalk_local[:19], "run": run.name,
             "direction": r.vehicle_direction, "approx_speed_mph": _num(r.approx_speed_mph),
             "pedestrian_was": r.pedestrian_was, "severity_why": r.severity_why,
             "scored_from": "tracks", "clip": (run / r.clip).resolve() if r.clip else None}
            for r in ev[ev["outcome"] == "did_not_yield"].itertuples()]
    return rows, updated


def from_log(run: Path) -> list[dict]:
    """Drivers who did not yield in a live session that never finished, from its yield_log.csv."""
    log = pd.read_csv(run / "yield_log.csv", dtype=str, keep_default_na=False)
    rows = []
    for r in log[log["outcome"] == "did_not_yield"].itertuples():
        phase = "in_crosswalk" if "in the crosswalk" in r.what else "waiting_at_curb"
        speed = re.search(r"about (\d+) mph", r.what)
        speed = float(speed.group(1)) if speed else None
        if getattr(r, "severity", ""):
            score, why, source = int(r.severity), r.severity_why, "live"
        else:
            score, why = severity(phase, None, speed, None, None)
            source = "log (waiting or in crosswalk, and speed only)"
        rows.append({"severity": score, "when": r.time_local[:19].replace("T", " "), "run": run.name,
                     "direction": r.direction, "approx_speed_mph": speed, "pedestrian_was": phase,
                     "severity_why": why, "scored_from": source,
                     "clip": (run.parent / "did_not_yield" / r.clip).resolve() if r.clip else None})
    return rows


def renames(rows: list[dict]) -> dict[Path, Path]:
    """Each clip that exists and doesn't start with its worst driver's score, and its new path."""
    worst: dict[Path, int] = {}
    for r in rows:
        if r["clip"] is not None:
            worst[r["clip"]] = max(worst.get(r["clip"], 0), r["severity"])
    out = {}
    for clip, score in worst.items():
        new = clip.with_name(f"S{score}_{SCORE_PREFIX.sub('', clip.name)}")
        if new != clip and clip.exists():
            out[clip] = new
    return out


def rename_in(csv: Path, names: dict[str, str]) -> None:
    """Point a yield_events.csv or yield_log.csv at renamed clips."""
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    if "clip" not in df.columns:
        return
    new = df["clip"].map(lambda c: str(Path(c).with_name(names[Path(c).name])) if Path(c).name in names else c)
    if not new.equals(df["clip"]):
        df.assign(clip=new).to_csv(csv, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("out"), help="folder holding the runs")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--calibration", type=Path, default=Path("calibration.yaml"),
                    help="used for a run that doesn't name its calibration file")
    ap.add_argument("--top", type=int, default=10, help="how many of the worst drivers to join in the reel")
    ap.add_argument("--dry-run", action="store_true", help="print the renames, change nothing")
    args = ap.parse_args()

    cfg = load_config(args.config)
    args.out = args.out.resolve()
    rows, rescored, busy = [], {}, set()
    for run in sorted(p for p in args.out.iterdir() if p.is_dir()):
        if (run / "yield_events.csv").exists() and (run / "tracks.pkl").exists():
            found, updated = from_events(run, cfg, args.calibration)
            if updated is not None:
                rescored[run / "yield_events.csv"] = updated
        elif (run / "yield_log.csv").exists():
            found = from_log(run)
            if time.time() - (run / "yield_log.csv").stat().st_mtime < RUNNING_SECONDS:
                busy.update(r["clip"] for r in found if r["clip"] is not None)
        else:
            continue
        rows.extend(found)

    moves = renames(rows)
    skipped = {old for old in moves if old in busy}
    moves = {old: new for old, new in moves.items() if old not in skipped}
    for old, new in moves.items():
        print(f"{'Would rename' if args.dry_run else 'Renamed'} {old.relative_to(args.out)} -> {new.name}")
    if skipped:
        print(f"Left {len(skipped)} clip(s) of a session that may still be running as they are. "
              "Run this again once it has stopped.")
    known = {r["clip"] for r in rows}
    unknown = [p for p in sorted((args.out / "did_not_yield").glob("*.mp4")) if p not in known]
    if unknown:
        print(f"{len(unknown)} clip(s) in did_not_yield/ aren't in any session's log, so they can't be "
              "scored: " + ", ".join(p.name for p in unknown))
    if args.dry_run:
        return

    for csv, df in rescored.items():
        df.to_csv(csv, index=False)
    for old, new in moves.items():
        old.rename(new)
    names = {old.name: new.name for old, new in moves.items()}
    if names:
        for csv in [*args.out.glob("*/yield_events.csv"), *args.out.glob("*/yield_log.csv")]:
            if not any(r["clip"] in skipped for r in rows if r["run"] == csv.parent.name):
                rename_in(csv, names)
    for r in rows:
        r["clip"] = moves.get(r["clip"], r["clip"])

    ranked = pd.DataFrame(rows, columns=COLUMNS).sort_values(
        ["severity", "approx_speed_mph", "when"], ascending=[False, False, True], na_position="last")
    ranked["clip"] = ranked["clip"].map(lambda c: str(c.relative_to(args.out)) if c is not None else "")
    ranked.to_csv(args.out / "worst_drivers.csv", index=False)
    print(f"{len(ranked)} drivers who did not yield, by score: "
          + ", ".join(f"{k}: {int((ranked['severity'] == k).sum())}" for k in range(5, 0, -1)))
    print(f"Wrote {args.out / 'worst_drivers.csv'}")

    worst = [args.out / c for c in dict.fromkeys(ranked["clip"]) if c and (args.out / c).exists()][:args.top]
    if worst:
        joined = args.out / "worst_did_not_yield.mp4"
        reel(worst, joined)
        print(f"Wrote {joined} ({len(worst)} clips, worst first). Watch each before sharing it.")


if __name__ == "__main__":
    main()
