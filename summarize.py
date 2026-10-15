"""Rebuild summary.md, summary.json and the charts for a finished run, without re-processing video.

Use it after editing config.yaml (speed limit, crosswalk length, ...) or after tape-measuring the
speed trap in calibration.yaml. If the trap distance changed, the speeds in vehicles.csv are
rescaled to match (speed is distance / time, so they scale exactly). The annotated video is not
re-rendered, so its on-screen speed labels keep the old values.

    uv run python summarize.py out/VIDEO
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from traffic.config import load_calibration, load_config
from traffic.report import write_summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path, help="folder written by analyze.py")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--calibration", type=Path, default=Path("calibration.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    cal = load_calibration(args.calibration)
    trap = cal["speed_trap"]
    meta_path = args.out_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text())
    used = meta["calibration"]
    old_ft, new_ft = float(used["distance_ft"]), float(trap["distance_ft"])

    if abs(new_ft - old_ft) > 1e-6:
        factor = new_ft / old_ft
        csv = args.out_dir / "vehicles.csv"
        veh = pd.read_csv(csv, keep_default_na=False, dtype=str)
        speed = pd.to_numeric(veh["speed_mph"], errors="coerce") * factor
        veh["speed_mph"] = speed.round(1).astype(object).where(speed.notna(), "")
        # Re-check the plausibility bounds; other discard reasons don't depend on distance.
        s = cfg["speed"]
        recheck = speed.notna() & veh["discard_reason"].isin(["", "implausible_speed"])
        ok = speed.between(float(s["min_mph"]), float(s["max_mph"]))
        veh.loc[recheck, "speed_valid"] = ok[recheck].map({True: "True", False: "False"})
        veh.loc[recheck, "discard_reason"] = ok[recheck].map({True: "", False: "implausible_speed"})
        veh.to_csv(csv, index=False)
        used["speeds_rescaled_from_ft"] = used.get("speeds_rescaled_from_ft", old_ft)
        print(f"Rescaled speeds by {factor:.4f} ({old_ft:g} ft -> {new_ft:g} ft). "
              "The annotated video still shows the old labels.")
    used.update({"distance_ft": new_ft,
                 "distance_measured": bool(trap.get("distance_measured")),
                 "distance_notes": trap.get("distance_notes", "")})
    meta["speed_limit_mph"] = float(cfg["speed_limit_mph"])
    meta["gate_labels"] = cal.get("gate_labels", {})
    meta["site_name"] = cal.get("site_name")
    meta_path.write_text(json.dumps(meta, indent=2))

    s = write_summary(args.out_dir, cfg)
    print(f"Wrote {args.out_dir / 'summary.md'}: {s['vehicles']['counted_total']} vehicles, "
          f"85th-percentile speed {s['speed'].get('p85_mph', 'n/a')} mph, "
          f"average pedestrian wait {s['crossing_gaps']['avg_wait_s']} s")


if __name__ == "__main__":
    main()
