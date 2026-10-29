"""CSV, JSON, markdown and chart outputs.

`write_summary(out_dir)` rebuilds summary.md/summary.json/charts purely from the
files in `out_dir`, so validate.py can re-run it after adding validation results.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from . import benchmarks as bm
from .config import load_config

TIME_FMT = "%Y-%m-%d %H:%M:%S"

# Reference categorical palette (validated light-mode slots 1-2), text inks, surface.
SERIES = ["#2a78d6", "#eb6834"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def local(start: datetime, pts: float | None) -> str:
    if pts is None:
        return ""
    t = start + timedelta(seconds=pts)
    return t.strftime(TIME_FMT) + f".{int(t.microsecond / 1e5)}"


def vehicle_rows(analyzer) -> pd.DataFrame:
    rows = []
    for tr in sorted(analyzer.vehicle_records, key=lambda t: t.count_pts or t.first_pts):
        valid, reason = analyzer.speed_status(tr)
        rows.append({
            "track_id": f"{'B' if tr.kind == 'bicycle' else 'V'}{tr.tid}",
            "class": tr.cls,
            "timestamp_local": local(analyzer.start_time, tr.count_pts or tr.first_pts),
            "video_seconds": round(tr.count_pts or tr.first_pts, 2),
            "counted": tr.count_pts is not None,
            "direction": analyzer.direction_of(tr),
            "entry_gate": tr.entry or "",
            "exit_gate": tr.exit or "",
            "movement": tr.movement,
            "movement_inferred": tr.inferred,
            "speed_mph": round(tr.speed_mph, 1) if tr.speed_mph is not None else None,
            "n_points": tr.n_trap_points,
            "speed_valid": valid,
            "discard_reason": reason,
            "seconds_tracked": round(tr.last_pts - tr.first_pts, 2),
        })
    return pd.DataFrame(rows, columns=[
        "track_id", "class", "timestamp_local", "video_seconds", "counted", "direction",
        "entry_gate", "exit_gate", "movement", "movement_inferred", "speed_mph", "n_points",
        "speed_valid", "discard_reason", "seconds_tracked"])


def ped_rows(analyzer) -> pd.DataFrame:
    rows = [{
        "track_id": f"P{tr.tid}",
        "timestamp_local": local(analyzer.start_time, tr.crossed_pts),
        "video_seconds": round(tr.crossed_pts, 2),
        "direction": tr.direction,
        "seconds_in_crosswalk": round(tr.inside_seconds, 1),
    } for tr in sorted(analyzer.ped_records, key=lambda t: t.crossed_pts)]
    return pd.DataFrame(rows, columns=["track_id", "timestamp_local", "video_seconds",
                                       "direction", "seconds_in_crosswalk"])


def ped_review_rows(analyzer) -> pd.DataFrame:
    """Everyone whose feet came near the crosswalk, counted or not, with the reason."""
    rows = [{
        "track_id": f"P{tr.tid}",
        "first_seen_local": local(analyzer.start_time, tr.first_pts),
        "video_seconds": round(tr.first_pts, 2),
        "seconds_tracked": round(tr.last_pts - tr.first_pts, 1),
        "reason": reason,
        "seconds_in_crosswalk": round(tr.inside_seconds, 1),
        "fraction_across": round((tr.max_proj - tr.min_proj) / analyzer.cw_axis_len, 2)
        if tr.first_inside_pts is not None else 0.0,
    } for tr, reason in sorted(analyzer.ped_reviewed, key=lambda x: x[0].first_pts)]
    return pd.DataFrame(rows, columns=["track_id", "first_seen_local", "video_seconds",
                                       "seconds_tracked", "reason", "seconds_in_crosswalk",
                                       "fraction_across"])


# --------------------------------------------------------------------------- summary
def _pct(n, d) -> float | None:
    return round(100.0 * n / d, 1) if d else None


def _fmt(v, suffix="") -> str:
    return "n/a" if v is None else f"{v}{suffix}"


def compute_summary(out_dir: Path, cfg: dict) -> dict:
    meta = json.loads((out_dir / "run_meta.json").read_text())
    veh = pd.read_csv(out_dir / "vehicles.csv", keep_default_na=False,
                      dtype={"speed_mph": str})
    veh["speed_mph"] = pd.to_numeric(veh["speed_mph"], errors="coerce")
    for col in ("counted", "speed_valid", "movement_inferred"):
        veh[col] = veh[col].astype(str).str.lower() == "true"
    peds = pd.read_csv(out_dir / "pedestrians.csv", keep_default_na=False)
    limit = meta["speed_limit_mph"]

    start = datetime.fromisoformat(meta["window_start"])
    end = datetime.fromisoformat(meta["window_end"])
    # Per-hour rates count only the time something was watched.
    not_watched = sum(g["minutes"] for g in meta.get("not_watched", [])) * 60
    hours = ((end - start).total_seconds() - not_watched) / 3600

    motor = veh[veh["class"] != "bicycle"]
    counted = motor[motor["counted"]]
    bikes = veh[(veh["class"] == "bicycle") & veh["counted"]]

    # Crossing gaps: when could a pedestrian get across both lanes?
    road = cfg["road"]
    critical_gap = road["crosswalk_length_ft"] / road["walking_speed_fps"] + road["startup_seconds"]
    counted_t = pd.to_datetime(counted["timestamp_local"])
    window_s = (end - start).total_seconds()
    gaps = bm.gap_analysis((counted_t - start.replace(tzinfo=None)).dt.total_seconds(),
                           window_s, critical_gap)
    arrivals, delays = gaps.pop("_arrivals"), gaps.pop("_delays")
    gaps.update({"crosswalk_length_ft": road["crosswalk_length_ft"],
                 "crosswalk_length_measured": bool(road.get("crosswalk_length_measured")),
                 "walking_speed_fps": road["walking_speed_fps"],
                 "startup_seconds": road["startup_seconds"]})

    # 15-minute clock bins
    bins = []
    b0 = start.replace(minute=start.minute - start.minute % 15, second=0, microsecond=0)
    b = b0
    while b < end:
        b1 = b + timedelta(minutes=15)
        sel = counted[(counted_t >= b.replace(tzinfo=None)) & (counted_t < b1.replace(tzinfo=None))]
        covered = (min(b1, end) - max(b, start)).total_seconds() / 60
        for g in meta.get("not_watched", []):
            g0, g1 = datetime.fromisoformat(g["start"]), datetime.fromisoformat(g["end"])
            covered -= max(0.0, (min(b1, g1) - max(b, g0)).total_seconds() / 60)
        lo, hi = (b - start).total_seconds(), (b1 - start).total_seconds()
        in_bin = delays[(arrivals >= lo) & (arrivals < hi)]
        bins.append({"start": b.strftime("%H:%M"), "end": b1.strftime("%H:%M"),
                     "minutes_recorded": round(covered, 1), "total": len(sel),
                     **{d: int((sel["direction"] == d).sum()) for d in meta["direction_labels"]},
                     "avg_ped_wait_s": round(float(in_bin.mean()), 1) if len(in_bin) else None})
        b = b1

    speeds = motor[motor["speed_valid"]]["speed_mph"]
    speed = {"n": int(len(speeds))}
    if len(speeds):
        speed.update({
            "mean_mph": round(float(speeds.mean()), 1),
            "median_mph": round(float(speeds.median()), 1),
            "p85_mph": round(float(speeds.quantile(0.85)), 1),
            "max_mph": round(float(speeds.max()), 1),
            "pct_over_limit": _pct(int((speeds > limit).sum()), len(speeds)),
            "pct_over_limit_plus_10": _pct(int((speeds > limit + 10).sum()), len(speeds)),
        })
    speed["by_direction"] = {
        d: {"n": int(len(s)), "median_mph": round(float(s.median()), 1) if len(s) else None,
            "p85_mph": round(float(s.quantile(0.85)), 1) if len(s) else None}
        for d, s in motor[motor["speed_valid"]].groupby("direction")["speed_mph"]}

    summary = {
        "recording": {"source": meta["source"], "start": meta["window_start"],
                      "end": meta["window_end"], "duration_minutes": round(hours * 60, 1),
                      "not_watched": meta.get("not_watched", []),
                      "start_time_source": meta["start_time_source"]},
        "vehicles": {
            "counted_total": int(len(counted)),
            "per_hour": round(len(counted) / hours) if hours else None,
            "by_direction": counted["direction"].value_counts().to_dict(),
            "by_class": counted["class"].value_counts().to_dict(),
            "by_movement": counted["movement"].value_counts().to_dict(),
            "movements": _movement_table(counted, meta.get("gate_labels") or {}),
            "all_tracked_by_movement": motor["movement"].value_counts().to_dict(),
            "bins_15min": bins,
        },
        "bicycles": {"counted_total": int(len(bikes)),
                     "by_direction": bikes["direction"].value_counts().to_dict()},
        "speed": {"limit_mph": limit, **speed,
                  "discards": motor[~motor["speed_valid"]]["discard_reason"].value_counts().to_dict()},
        "pedestrians_reviewed": _ped_review_summary(out_dir),
        "yielding": _yield_summary(out_dir, len(peds)),
        "pedestrians": {"crossings": int(len(peds)),
                        "per_hour": round(len(peds) / hours, 1) if hours else None,
                        "by_direction": peds["direction"].value_counts().to_dict() if len(peds) else {}},
        "site_name": meta.get("site_name"),
        "crossing_gaps": gaps,
        "calibration": meta["calibration"],
        "processing": meta["processing"],
        "validation": None,
    }
    summary["benchmarks"] = _benchmarks(summary, cfg)
    vpath = out_dir / "validation.json"
    if vpath.exists():
        summary["validation"] = json.loads(vpath.read_text())
    return summary


def _yield_summary(out_dir: Path, crossings: int) -> dict | None:
    path = out_dir / "yield_events.csv"
    if not path.exists():
        return None
    ev = pd.read_csv(path, keep_default_na=False)
    peds = pd.read_csv(out_dir / "pedestrians.csv")
    outcome = ev["outcome"] if len(ev) else pd.Series(dtype=str)
    yielded = int((outcome == "yielded").sum())
    missed = ev[outcome == "did_not_yield"] if len(ev) else ev
    judged = yielded + len(missed)
    waits = pd.to_numeric(peds.get("wait_at_curb_s", pd.Series(dtype=float)), errors="coerce").dropna()
    return {
        "crossings": crossings,
        "crossings_with_approaching_vehicle": _crossings_with(peds, ["drivers_did_not_yield", "drivers_yielded",
                                                                     "too_close_to_stop"]),
        "crossings_with_driver_not_yielding": _crossings_with(peds, ["drivers_did_not_yield"]),
        "yielded": yielded,
        "did_not_yield": int(len(missed)),
        "did_not_yield_in_crosswalk": int((missed["pedestrian_was"] == "in_crosswalk").sum()) if len(missed) else 0,
        "did_not_yield_waiting": int((missed["pedestrian_was"] == "waiting_at_curb").sum()) if len(missed) else 0,
        "too_close_to_stop": int((outcome == "too_close_to_stop").sum()),
        "yield_rate_pct": round(100.0 * yielded / judged, 1) if judged else None,
        "avg_wait_at_curb_s": round(float(waits.mean()), 1) if len(waits) else None,
        "max_wait_at_curb_s": round(float(waits.max()), 1) if len(waits) else None,
        "clips": int((missed["clip"] != "").sum()) if len(missed) else 0,
        "note_groups": bool(len(ev)) and ev["pedestrian"].astype(str).str.contains("\\+").any(),
        "by_direction": {
            d: {"yielded": int((g["outcome"] == "yielded").sum()),
                "did_not_yield": int((g["outcome"] == "did_not_yield").sum()),
                "too_close_to_stop": int((g["outcome"] == "too_close_to_stop").sum())}
            for d, g in ev.groupby("vehicle_direction")} if len(ev) else {},
    }


def _crossings_with(peds: pd.DataFrame, cols: list[str]) -> int:
    if not all(c in peds.columns for c in cols):
        return 0
    return int((peds[cols].fillna(0).sum(axis=1) > 0).sum())


def _yield_markdown(y: dict, limit: float) -> list[str]:
    L = ["## Drivers yielding to pedestrians\n"]
    if not y["crossings"]:
        return L + ["No pedestrian crossings were recorded.\n"]
    L.append(f"- **{y['crossings_with_driver_not_yielding']} of {y['crossings']} crossings** had at least "
             "one driver who did not yield.")
    if y["yield_rate_pct"] is not None:
        L.append(f"- **Driver yield rate: {y['yield_rate_pct']}%** ({y['yielded']} yielded, "
                 f"{y['did_not_yield']} did not).")
    L.append(f"  - Drove through while the pedestrian was **in the crosswalk**: {y['did_not_yield_in_crosswalk']}")
    L.append(f"  - Drove past a pedestrian **waiting at the curb**: {y['did_not_yield_waiting']}")
    judged = y["yielded"] + y["did_not_yield"]
    if judged < 10:
        L.append(f"- **Small sample:** only {judged} driver(s) met a pedestrian in this recording, too few "
                 "for a reliable yield rate. Record more sessions, ideally when more people walk.")
    if y.get("note_groups"):
        L.append("- When people cross together, each driver is judged once for the group.")
    L.append(f"- Arrived too soon after the pedestrian appeared to be expected to stop (not counted above): "
             f"{y['too_close_to_stop']}")
    if y["avg_wait_at_curb_s"] is not None:
        L.append(f"- Pedestrians waited at the curb {y['avg_wait_at_curb_s']} s on average before stepping "
                 f"out (longest {y['max_wait_at_curb_s']} s).")
    if y.get("by_direction"):
        L.append("\n| Drivers by direction | Yielded | Did not yield | Too close to stop | Yield rate |")
        L.append("|---|---|---|---|---|")
        for d, r in y["by_direction"].items():
            n = r["yielded"] + r["did_not_yield"]
            rate = f"{100 * r['yielded'] / n:.0f}%" if n else "n/a"
            L.append(f"| {d.capitalize()} | {r['yielded']} | {r['did_not_yield']} | {r['too_close_to_stop']} | {rate} |")
        L.append("")
    if y["clips"]:
        L.append(f"- Video clip of each driver not yielding: `yield_clips/` ({y['clips']} clips, also joined "
                 "in `yield_clips/all_did_not_yield.mp4`). Each one should be watched before it's shared.")
    L.append("")
    L.append("How this is judged: a pedestrian is \"trying to cross\" from when they wait at the curb at "
             "either end of the crosswalk until they finish crossing. A driver **did not yield** if they "
             "reached the crosswalk in that window, in a lane the pedestrian still had to cross, without "
             "slowing below 5 mph, and at least 3 seconds after the pedestrian appeared (enough time to "
             f"react and stop at {limit:g} mph). Only the first driver per direction who stops counts as "
             "yielding, so a queue behind them doesn't inflate the rate. California Vehicle Code 21950 "
             "requires drivers to yield to pedestrians crossing within a marked crosswalk. For context, "
             "FHWA reports yield rates with RRFBs from 19% to as high as 98%.\n")
    return L


def _ped_review_summary(out_dir: Path) -> dict | None:
    path = out_dir / "pedestrians_reviewed.csv"
    if not path.exists():
        return None
    r = pd.read_csv(path)
    return r["reason"].value_counts().to_dict()


def _movement_table(counted: pd.DataFrame, labels: dict) -> list[dict]:
    def name(gate: str) -> str:
        return labels.get(gate, gate.capitalize()) if gate else "not seen"

    rows = (counted.groupby(["entry_gate", "exit_gate", "movement"]).size()
            .sort_values(ascending=False).reset_index(name="vehicles"))
    return [{"from": name(r.entry_gate), "to": name(r.exit_gate), "movement": r.movement,
             "vehicles": int(r.vehicles)} for r in rows.itertuples()]


def _benchmarks(s: dict, cfg: dict) -> dict:
    vph = s["vehicles"]["per_hour"] or 0
    lo_share, hi_share = bm.PEAK_HOUR_SHARE
    daily_mid = vph / 0.10
    out = {
        "daily_estimate": [int(round(vph / hi_share, -2)), int(round(vph / lo_share, -2))],
        "daily_midpoint": int(round(daily_mid, -2)),
        "functional_class": bm.functional_class(daily_mid),
        "short_window": s["recording"]["duration_minutes"] < 50,
        "existing_treatments": list(cfg["road"].get("existing_treatments") or []),
    }
    limit = s["speed"]["limit_mph"]
    p85 = s["speed"].get("p85_mph")
    if int(cfg["road"]["lanes"]) == 2:
        band = bm.aadt_band(daily_mid)
        posted = bm.speed_band(limit)
        out["step"] = {"aadt_band": band, "posted_band": posted,
                       "posted_cell": bm.STEP_TABLE_2_LANES[band][posted]}
        if p85 is not None:
            op = bm.speed_band(p85)
            out["step"].update({"operating_band": op,
                                "operating_cell": bm.STEP_TABLE_2_LANES[band][op]})
    out["risk"] = {"at_limit": {"death_pct": round(bm.risk_at(limit, bm.DEATH_RISK_MPH)),
                                "severe_pct": round(bm.risk_at(limit, bm.SEVERE_INJURY_RISK_MPH))}}
    if p85 is not None:
        out["risk"]["at_p85"] = {"mph": p85,
                                 "death_pct": round(bm.risk_at(p85, bm.DEATH_RISK_MPH)),
                                 "severe_pct": round(bm.risk_at(p85, bm.SEVERE_INJURY_RISK_MPH))}

    by_dir = s["speed"].get("by_direction") or {}
    out["sight_distance"] = []
    for direction, conf in (cfg.get("sight_distance") or {}).items():
        conf = conf or {}
        grade = float(conf.get("grade_pct") or 0)
        feet = conf.get("feet")
        p85_dir = (by_dir.get(direction) or {}).get("p85_mph")
        needed = bm.design_stopping_sight_distance_ft(limit, grade)
        out["sight_distance"].append({
            "direction": direction,
            "feet": float(feet) if feet is not None else None,
            "grade_pct": grade,
            "notes": conf.get("notes") or "",
            "needed_at_limit_ft": needed,
            "p85_mph": p85_dir,
            "needed_at_p85_ft": bm.design_stopping_sight_distance_ft(p85_dir, grade) if p85_dir else None,
            "shortfall_ft": (max(0, needed - float(feet)) if feet is not None else None),
        })
    return out


def _charts(out_dir: Path, s: dict, veh_speeds: pd.Series, directions: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def style(ax):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK_2)
        ax.tick_params(colors=INK_2, length=0)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    # Vehicles per 15 minutes, stacked by direction
    # Periods with only a few minutes of video (the recording started or ended mid-period)
    # look misleadingly empty, so leave them out of the chart; the table still lists them.
    all_bins = s["vehicles"]["bins_15min"]
    bins = [b for b in all_bins if b["minutes_recorded"] >= 5]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150, facecolor=SURFACE)
    style(ax)
    x = range(len(bins))
    bottom = [0] * len(bins)
    for i, d in enumerate(directions):
        vals = [b.get(d, 0) for b in bins]
        ax.bar(x, vals, bottom=bottom, width=0.6, color=SERIES[i % len(SERIES)], label=d,
               edgecolor=SURFACE, linewidth=2)
        bottom = [a + v for a, v in zip(bottom, vals)]
    for xi, b in zip(x, bins):
        partial = b["minutes_recorded"] < 15
        ax.text(xi, b["total"] + max(bottom) * 0.02, f"{b['total']}" + ("*" if partial else ""),
                ha="center", va="bottom", color=INK, fontsize=10)
    ax.set_xticks(list(x), [f"{b['start']}" for b in bins], color=INK_2)
    ax.set_ylabel("Vehicles", color=INK_2)
    ax.set_title("Vehicles per 15 minutes", color=INK, loc="left", fontsize=13)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=len(directions))
    notes = []
    if any(b["minutes_recorded"] < 15 for b in bins):
        notes.append("* partial 15-minute period")
    if len(bins) < len(all_bins):
        notes.append("periods with under 5 minutes of video not shown")
    if notes:
        fig.text(0.99, 0.01, "; ".join(notes), ha="right", color=INK_2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "vehicles_per_15min.png", facecolor=SURFACE)
    plt.close(fig)

    # Speed histogram
    if len(veh_speeds):
        limit, p85 = s["speed"]["limit_mph"], s["speed"]["p85_mph"]
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150, facecolor=SURFACE)
        style(ax)
        lo = int(veh_speeds.min() // 2 * 2)
        hi = int(veh_speeds.max() // 2 * 2 + 4)
        ax.hist(veh_speeds, bins=range(lo, hi, 2), color=SERIES[0], edgecolor=SURFACE, linewidth=2)
        top = ax.get_ylim()[1]
        # Limit label to the left of its line, 85th percentile to the right, so they never collide.
        for v, label, ha, dx, y in [(limit, f"Limit {limit:g} mph", "right", -0.4, 0.97),
                                    (p85, f"85th percentile {p85:g} mph", "left", 0.4, 0.88)]:
            ax.axvline(v, color=INK, linewidth=1.2, linestyle="--")
            ax.text(v + dx, top * y, label, color=INK, fontsize=9, va="top", ha=ha,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
        ax.set_xlabel("Speed (mph)", color=INK_2)
        ax.set_ylabel("Vehicles", color=INK_2)
        title = "Through-vehicle speeds"
        if not s["calibration"].get("distance_measured"):
            title += " (ESTIMATE: speed trap not yet tape-measured)"
        ax.set_title(title, color=INK, loc="left", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / "speed_histogram.png", facecolor=SURFACE)
        plt.close(fig)


def write_summary(out_dir: Path, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config(Path("config.yaml"))
    s = compute_summary(out_dir, cfg)
    (out_dir / "summary.json").write_text(json.dumps(s, indent=2, default=str))
    veh = pd.read_csv(out_dir / "vehicles.csv", keep_default_na=False, dtype={"speed_mph": str})
    valid = veh["speed_valid"].astype(str).str.lower() == "true"
    speeds = pd.to_numeric(veh[valid & (veh["class"] != "bicycle")]["speed_mph"], errors="coerce").dropna()
    meta = json.loads((out_dir / "run_meta.json").read_text())
    directions = list(meta["direction_labels"])
    _charts(out_dir, s, speeds, directions)
    (out_dir / "summary.md").write_text(_markdown(s, directions))
    return s


def _markdown(s: dict, directions: list[str]) -> str:
    r, v, sp, p, cal = s["recording"], s["vehicles"], s["speed"], s["pedestrians"], s["calibration"]
    start = datetime.fromisoformat(r["start"])
    end = datetime.fromisoformat(r["end"])
    lim = sp["limit_mph"]
    L = []
    L.append(f"# Traffic study: {s.get('site_name') or 'crosswalk'}\n")
    L.append(f"**Recording:** {start:%a %b %d %Y}, {start:%H:%M:%S} to {end:%H:%M:%S} PT "
             f"({r['duration_minutes']} minutes). Start time from {r['start_time_source']}.\n")
    if r.get("not_watched"):
        gaps = "; ".join(f"{datetime.fromisoformat(g['start']):%H:%M}–"
                         f"{datetime.fromisoformat(g['end']):%H:%M} ({g['minutes']:g} min)"
                         for g in r["not_watched"])
        L.append(f"**Not watched** (computer asleep or camera offline): {gaps}. The minutes above "
                 f"and the per-hour rates leave these out.\n")

    L.append("## Vehicles through the crosswalk\n")
    L.append(f"- **{v['counted_total']} motor vehicles** crossed the count line "
             f"(**{_fmt(v['per_hour'])} per hour**).")
    for d in directions:
        L.append(f"  - {d.capitalize()}: {v['by_direction'].get(d, 0)}")
    L.append(f"- Bicycles (counted separately): {s['bicycles']['counted_total']}")
    if v["by_class"]:
        L.append("- By type: " + ", ".join(f"{k} {n}" for k, n in v["by_class"].items()))
    L.append("")
    L.append("| 15 min starting | " + " | ".join(d.capitalize() for d in directions)
             + " | Total | Avg pedestrian wait | Minutes recorded |")
    L.append("|---|" + "---|" * (len(directions) + 3))
    for b in v["bins_15min"]:
        L.append(f"| {b['start']} | " + " | ".join(str(b.get(d, 0)) for d in directions)
                 + f" | {b['total']} | {_fmt(b.get('avg_ped_wait_s'), ' s')} | {b['minutes_recorded']} |")
    L.append("\n![Vehicles per 15 minutes](vehicles_per_15min.png)\n")

    L.append("### Where counted vehicles came from and went\n")
    L.append("| From | To | Movement | Vehicles |\n|---|---|---|---|")
    for m in v["movements"]:
        L.append(f"| {m['from']} | {m['to']} | {m['movement'].replace('_', ' ')} | {m['vehicles']} |")
    L.append("\n\"not seen\" means the vehicle was not tracked across that entry or exit (for example, "
             "hidden behind a larger vehicle). Every row is still counted in the total.\n")

    L.append("## Speeds\n")
    if not cal.get("distance_measured"):
        L.append("> **ESTIMATES ONLY.** The speed-trap distance "
                 f"({cal['distance_ft']} ft) is a guess and has not been tape-measured. "
                 "Every speed scales directly with it: if the true distance is 10% longer, "
                 "every speed is 10% higher.\n")
    if sp["n"]:
        L.append(f"Through vehicles with a valid speed: **{sp['n']}**\n")
        L.append("| Statistic | Value |\n|---|---|")
        L.append(f"| Mean | {sp['mean_mph']} mph |")
        L.append(f"| Median | {sp['median_mph']} mph |")
        L.append(f"| 85th percentile | **{sp['p85_mph']} mph** |")
        L.append(f"| Max | {sp['max_mph']} mph |")
        L.append(f"| Over {lim} mph limit | {sp['pct_over_limit']}% |")
        L.append(f"| Over {lim + 10} mph | {sp['pct_over_limit_plus_10']}% |")
        if sp["by_direction"]:
            L.append("\n| Direction | Vehicles | Median | 85th pct |\n|---|---|---|---|")
            for d, ds in sp["by_direction"].items():
                L.append(f"| {d} | {ds['n']} | {_fmt(ds['median_mph'], ' mph')} | {_fmt(ds['p85_mph'], ' mph')} |")
        L.append("\n![Speed histogram](speed_histogram.png)\n")
    else:
        L.append("No valid speeds were measured.\n")
    if sp["discards"]:
        L.append("Vehicles without a speed, by reason: "
                 + ", ".join(f"{k} {n}" for k, n in sp["discards"].items()) + ".\n")

    L.append("## Pedestrians\n")
    L.append(f"- **{p['crossings']} crossings** ({_fmt(p['per_hour'])} per hour)")
    for d, n in p["by_direction"].items():
        L.append(f"  - {d.capitalize()}: {n}")
    rv = s.get("pedestrians_reviewed")
    if rv:
        others = {k: n for k, n in rv.items() if k != "counted"}
        if others:
            L.append(f"- {sum(others.values())} other people came near the crosswalk but were not counted: "
                     + ", ".join(f"{k.replace('_', ' ')} {n}" for k, n in others.items())
                     + ". Their paths are in `ped_check/` for review.")
    L.append("- Driver yielding behavior is not yet measured.\n")

    g = s["crossing_gaps"]
    L.append("### Waiting for a safe gap\n")
    L.append(f"Crossing both lanes takes about **{g['critical_gap_s']} seconds** "
             f"({g['crosswalk_length_ft']} ft at {g['walking_speed_fps']} ft/s, plus "
             f"{g['startup_seconds']:g} s to start)"
             + ("" if g["crosswalk_length_measured"] else ", using an estimated crosswalk length")
             + ". Measured from the gaps between vehicles passing the crosswalk:\n")
    L.append("| Measure | Value |\n|---|---|")
    L.append(f"| Gaps between vehicles long enough to cross | {_fmt(g['pct_gaps_long_enough'], '%')} |")
    L.append(f"| Pedestrians who must wait (arriving at a random moment) | {_fmt(g['pct_must_wait'], '%')} |")
    L.append(f"| Average wait | **{_fmt(g['avg_wait_s'], ' s')}** (HCM level of service **{g['los']}**) |")
    L.append(f"| 1 in 10 pedestrians waits longer than | {_fmt(g['p90_wait_s'], ' s')} |")
    L.append(f"| Longest wait | {_fmt(g['max_wait_s'], ' s')} |")
    L.append(f"\nIf the same number of vehicles arrived at random, the average wait would be "
             f"{g['random_traffic_avg_wait_s']} s.\n")

    if s.get("yielding"):
        L.extend(_yield_markdown(s["yielding"], s["speed"]["limit_mph"]))
    L.extend(_benchmark_markdown(s))
    L.append("## Method and calibration\n")
    L.append(f"- Detection: {s['processing']['model']} (YOLO) on every {s['processing']['stride']} "
             f"frame(s); tracking: ByteTrack.")
    L.append("- A vehicle is counted when its ground contact point crosses the count line, "
             "placed beside the crosswalk across both lanes.")
    L.append("- Speed = speed-trap distance / time between crossing trap line A and line B, using "
             "per-frame video timestamps. Only `through` vehicles are included; turning vehicles "
             "slow down by design.")
    L.append(f"- Speed-trap distance: {cal['distance_ft']} ft "
             f"({'tape-measured' if cal.get('distance_measured') else 'NOT measured, guess'}). "
             f"{cal.get('distance_notes') or ''}"
             + (f" Speeds were rescaled from the {cal['speeds_rescaled_from_ft']:g} ft used during "
                "processing; the annotated video shows the original values."
                if cal.get("speeds_rescaled_from_ft") else ""))
    L.append(f"- Calibration file dated {cal.get('created', 'unknown')}.\n")

    L.append("## Validation\n")
    val = s.get("validation")
    if not val:
        L.append("Validation has not been done yet. Counts have not been checked against a manual "
                 "count, and speeds have not been checked against a known-speed reference.\n")
    else:
        if val.get("count_check"):
            c = val["count_check"]
            L.append(f"- Manual count {c['window']}: manual {c['manual']}, automated {c['automated']} "
                     f"({c['pct_difference']:+.1f}%). Target: within 5%. "
                     f"**{'PASS' if abs(c['pct_difference']) <= 5 else 'FAIL'}**")
        if val.get("speed_check"):
            c = val["speed_check"]
            L.append(f"- Speed check against {c['n_matched']} reference readings: mean error "
                     f"{c['mean_error_mph']:+.1f} mph, mean absolute error {c['mae_mph']:.1f} mph, "
                     f"max error {c['max_abs_error_mph']:.1f} mph. Target MAE <= 3 mph. "
                     f"**{'PASS' if c['mae_mph'] <= 3 else 'FAIL'}**")
        L.append("")
    return "\n".join(L) + "\n"


SPEED_BAND_NAMES = {"<=30": "30 mph or less", "35": "35 mph", ">=40": "40 mph or more"}


def _sight_distance_markdown(rows: list[dict], lim: float) -> list[str]:
    if not rows:
        return []
    L = ["### Sight distance on the approaches\n",
         "How far back a driver can first see a pedestrian at the crosswalk, compared with the AASHTO "
         "stopping sight distance: the distance a driver needs to notice a hazard (2.5 seconds) and "
         "brake to a stop.\n",
         f"| Approach | Can first see a pedestrian | Needed at the {lim:g} mph limit | "
         "Needed at the measured 85th-percentile speed | Shortfall |",
         "|---|---|---|---|---|"]
    for r in rows:
        seen = f"{r['feet']:g} ft" if r["feet"] is not None else "**not yet measured**"
        p85 = (f"{r['needed_at_p85_ft']} ft ({r['p85_mph']} mph)" if r["needed_at_p85_ft"] else "n/a")
        if r["shortfall_ft"] is None:
            short = "n/a"
        else:
            short = f"**{r['shortfall_ft']:g} ft short**" if r["shortfall_ft"] > 0 else "none"
        grade = f", {abs(r['grade_pct']):g}% {'downhill' if r['grade_pct'] < 0 else 'uphill'}" if r["grade_pct"] else ""
        L.append(f"| {r['direction'].capitalize()}{grade} | {seen} | {r['needed_at_limit_ft']} ft | {p85} | {short} |")
    L.append("")
    for r in rows:
        if r["notes"]:
            L.append(f"- {r['direction'].capitalize()}: {r['notes']}")
    if any(r["notes"] for r in rows):
        L.append("")
    if any(r["feet"] is None for r in rows):
        L.append("To measure: one person stands at the curb at the crosswalk. Another walks back along "
                 "the sidewalk on that approach with their eyes about 3.5 ft above the road (a driver's "
                 "eye height, roughly crouching), until they can no longer see the first person's knees "
                 "(AASHTO assumes a 2 ft high object). The distance between them, by measuring wheel or "
                 "Google Maps, is the sight distance.\n")
    L.append("Where sight distance is shorter than needed, FHWA's rules for RRFBs (Interim Approval "
             "IA-21, condition 2d, now part of the 2023 MUTCD) allow an **additional RRFB on that approach "
             "in advance of the crosswalk**, with a pedestrian crossing sign and an AHEAD or distance "
             "plaque. FHWA's crosswalk guide (Table 2) lists RRFBs as addressing inadequate "
             "conspicuity/visibility and drivers not yielding. The MUTCD's HILL BLOCKS VIEW sign (W7-6) "
             "warns drivers approaching a crest with limited sight distance.\n")
    return L


def _benchmark_markdown(s: dict) -> list[str]:
    b, v, sp, g = s["benchmarks"], s["vehicles"], s["speed"], s["crossing_gaps"]
    names = bm.STEP_COUNTERMEASURES
    lim = sp["limit_mph"]
    L = ["## How this compares\n", "### Traffic volume\n"]
    lo, hi = b["daily_estimate"]
    L.append(f"{_fmt(v['per_hour'])} vehicles per hour is roughly **{lo:,} to {hi:,} vehicles per day**, "
             "assuming the busiest hour carries 8 to 12% of daily traffic"
             + (" (a rough estimate from less than an hour of video)" if b["short_window"] else "")
             + ". A 24-hour count would confirm it. FHWA's typical daily volumes for city streets:\n")
    L.append("| Street type | Typical vehicles/day | |\n|---|---|---|")
    for name, a, z in bm.FUNCTIONAL_CLASSES:
        mark = "**this street's estimate falls here**" if name in b["functional_class"] else ""
        L.append(f"| {name} | {a:,} to {z:,} | {mark} |")
    L.append("")

    if "step" in b:
        st = b["step"]
        L.append("### FHWA crossing treatments\n")
        L.append(f"FHWA's *Guide for Improving Pedestrian Safety at Uncontrolled Crossing Locations* "
                 f"(Table 1) suggests treatments by number of lanes, posted speed and daily traffic. "
                 f"For a 2-lane road with {st['aadt_band']} vehicles/day and a {lim:g} mph posted "
                 f"limit (the {SPEED_BAND_NAMES[st['posted_band']]} column), it lists:\n")
        have = set(b.get("existing_treatments", []))
        L.extend(f"- {names[n]}" + (" (**already in place**)" if n in have else "")
                 for n in st["posted_cell"])
        L.append("")
        if 7 not in st["posted_cell"]:
            L.append("An RRFB is not listed in this column. The guide says unlisted treatments are "
                     "generally not appropriate, but exceptions may be made on engineering judgment.\n")
        if st.get("operating_band") and st["operating_band"] != st["posted_band"]:
            added = [n for n in st["operating_cell"] if n not in st["posted_cell"]]
            L.append(f"The table is keyed to the **posted** limit, but traffic here moves faster: the "
                     f"measured 85th-percentile speed of {sp['p85_mph']} mph matches the "
                     f"{SPEED_BAND_NAMES[st['operating_band']]} column, which adds: "
                     + (", ".join(names[n] for n in added) if added else "nothing new") + ".\n")
        if not s["calibration"].get("distance_measured"):
            L.append("(Speeds are estimates until the speed trap is tape-measured.)\n")

    r = b["risk"]
    L.append("### Speed and injury risk\n")
    L.append("Average risk for a pedestrian struck by a car (AAA Foundation, Tefft 2011):\n")
    L.append("| Impact speed | Risk of death | Risk of severe injury |\n|---|---|---|")
    L.append(f"| {lim:g} mph (posted limit) | ~{r['at_limit']['death_pct']}% | ~{r['at_limit']['severe_pct']}% |")
    if "at_p85" in r:
        L.append(f"| {r['at_p85']['mph']} mph (measured 85th percentile) | ~{r['at_p85']['death_pct']}% "
                 f"| ~{r['at_p85']['severe_pct']}% |")
    L.append("\nThe risk climbs steeply: 10% at 23 mph, 25% at 32 mph, 50% at 42 mph. Older "
             "pedestrians face much higher risk: a 70-year-old struck at 25 mph has about the same "
             "risk as a 30-year-old struck at 35 mph.\n")

    L.extend(_sight_distance_markdown(b.get("sight_distance") or [], lim))
    L.append("### Crossing delay\n")
    L.append("The Highway Capacity Manual grades unsignalized crossings by average pedestrian wait: "
             "A under 5 s, B 5 to 10 s, C 10 to 20 s, D 20 to 30 s, E 30 to 45 s, F over 45 s. "
             f"This crossing measured **{_fmt(g['avg_wait_s'], ' s')} (grade {g['los']})**.\n")

    L.append("Sources: AASHTO, *A Policy on Geometric Design of Highways and Streets* (the Green Book), "
             "stopping sight distance. FHWA, MUTCD Interim Approval IA-21 for RRFBs (2018), now in the "
             "2023 MUTCD. FHWA, *Guide for Improving Pedestrian Safety at Uncontrolled Crossing "
             "Locations* (2018), Table 1. FHWA, *Highway Functional Classification Concepts, "
             "Criteria and Procedures* (2023). Tefft, *Impact Speed and a Pedestrian's Risk of Severe "
             "Injury or Death*, AAA Foundation for Traffic Safety (2011). Transportation Research "
             "Board, *Highway Capacity Manual*, pedestrian level of service at unsignalized crossings.\n")
    return L
