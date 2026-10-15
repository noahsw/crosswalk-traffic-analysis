"""Check automated results against manual observations, then refresh summary.md.

Manual count check (count vehicles by hand while watching the video for a window):
    uv run python validate.py out/VIDEO --count-window 08:40 08:50 --manual-count 37

Speed check against known speeds (radar readings, or your own car driven at a
steady speed on cruise control). CSV columns: timestamp, direction, mph
    timestamp: local time as HH:MM:SS (same day as the recording)
    direction: as in vehicles.csv, e.g. eastbound
    uv run python validate.py out/VIDEO --speeds reference_speeds.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from traffic.report import write_summary


def _at(day: datetime, hhmm: str) -> datetime:
    t = datetime.strptime(hhmm, "%H:%M:%S" if hhmm.count(":") == 2 else "%H:%M").time()
    return datetime.combine(day.date(), t)


def count_check(veh: pd.DataFrame, day: datetime, start: str, end: str, manual: int) -> dict:
    t0, t1 = _at(day, start), _at(day, end)
    ts = pd.to_datetime(veh["timestamp_local"])
    sel = veh[veh["counted"] & (veh["class"] != "bicycle") & (ts >= t0) & (ts < t1)]
    auto = int(len(sel))
    pct = 100.0 * (auto - manual) / manual if manual else float("nan")
    return {"window": f"{start}-{end}", "manual": manual, "automated": auto,
            "pct_difference": round(pct, 1)}


def speed_check(veh: pd.DataFrame, day: datetime, ref_csv: Path, tolerance: float) -> dict:
    ref = pd.read_csv(ref_csv)
    cand = veh[veh["counted"] & veh["speed_mph"].notna()].copy()
    cand["t"] = pd.to_datetime(cand["timestamp_local"])
    matches, unmatched = [], []
    for _, r in ref.iterrows():
        t = _at(day, str(r["timestamp"]))
        pool = cand[cand["direction"].str.lower() == str(r["direction"]).lower()]
        if pool.empty:
            unmatched.append(str(r["timestamp"]))
            continue
        dt = (pool["t"] - t).abs().dt.total_seconds()
        i = dt.idxmin()
        if dt[i] > tolerance:
            unmatched.append(str(r["timestamp"]))
            continue
        v = pool.loc[i]
        matches.append({"reference_time": str(r["timestamp"]), "track_id": v["track_id"],
                        "seconds_apart": round(float(dt[i]), 1),
                        "reference_mph": float(r["mph"]), "measured_mph": float(v["speed_mph"]),
                        "error_mph": round(float(v["speed_mph"]) - float(r["mph"]), 1)})
    errs = pd.Series([m["error_mph"] for m in matches], dtype=float)
    return {
        "n_reference": int(len(ref)), "n_matched": len(matches), "unmatched": unmatched,
        "mean_error_mph": round(float(errs.mean()), 2) if len(errs) else None,
        "mae_mph": round(float(errs.abs().mean()), 2) if len(errs) else None,
        "max_abs_error_mph": round(float(errs.abs().max()), 2) if len(errs) else None,
        "matches": matches,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path, help="folder written by analyze.py")
    ap.add_argument("--count-window", nargs=2, metavar=("START", "END"), help="local times, e.g. 08:40 08:50")
    ap.add_argument("--manual-count", type=int)
    ap.add_argument("--speeds", type=Path, help="CSV of reference speeds")
    ap.add_argument("--tolerance", type=float, default=5.0, help="max seconds between reference and track")
    args = ap.parse_args()

    meta = json.loads((args.out_dir / "run_meta.json").read_text())
    day = datetime.fromisoformat(meta["window_start"])
    veh = pd.read_csv(args.out_dir / "vehicles.csv", keep_default_na=False, dtype={"speed_mph": str})
    veh["speed_mph"] = pd.to_numeric(veh["speed_mph"], errors="coerce")
    veh["counted"] = veh["counted"].astype(str).str.lower() == "true"

    vpath = args.out_dir / "validation.json"
    val = json.loads(vpath.read_text()) if vpath.exists() else {}
    if args.count_window:
        if args.manual_count is None:
            ap.error("--count-window needs --manual-count")
        val["count_check"] = c = count_check(veh, day, *args.count_window, args.manual_count)
        print(f"Count {c['window']}: manual {c['manual']}, automated {c['automated']} "
              f"({c['pct_difference']:+.1f}%, target within 5%)")
    if args.speeds:
        val["speed_check"] = s = speed_check(veh, day, args.speeds, args.tolerance)
        print(f"Speeds: matched {s['n_matched']} of {s['n_reference']}; mean error "
              f"{s['mean_error_mph']} mph, MAE {s['mae_mph']} mph (target <= 3), max {s['max_abs_error_mph']}")
        for m in s["matches"]:
            print(f"  {m['reference_time']}  ref {m['reference_mph']:5.1f}  measured "
                  f"{m['measured_mph']:5.1f}  ({m['track_id']}, {m['seconds_apart']}s apart)")
        if s["unmatched"]:
            print("  No matching vehicle for: " + ", ".join(s["unmatched"]))
    if not val:
        ap.error("nothing to validate: pass --count-window/--manual-count and/or --speeds")
    vpath.write_text(json.dumps(val, indent=2))
    write_summary(args.out_dir)
    print(f"Updated {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
