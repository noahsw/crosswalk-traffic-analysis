"""Count vehicles and pedestrians, estimate speeds, and render an annotated video.

    uv run python analyze.py input/VIDEO.MOV
    uv run python analyze.py input/VIDEO.MOV --minutes 5                  # quick test: first 5 min
    uv run python analyze.py input/VIDEO.MOV --start-min 20 --minutes 5   # minutes 20-25
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from traffic import render
from traffic.config import load_calibration, load_config
from traffic.pipeline import Detector, new_analyzer, write_run
from traffic.report import write_summary
from traffic.video import FrameReader, VideoWriter, recording_start, strip_audio
from yielding import analyze_yields


def status(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--calibration", type=Path, default=Path("calibration.yaml"))
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder (default out/<video name>, or out/<video name>_<range> for partial runs)")
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--start-time", help='override recording start, local time, e.g. "2026-10-15T08:29:01"')
    ap.add_argument("--minutes", type=float, default=None,
                    help="only process this many minutes of video (for quick tests)")
    ap.add_argument("--start-min", type=float, default=0.0,
                    help="start this many minutes into the video (default 0)")
    ap.add_argument("--stride", type=int, default=None, help="detect every Nth frame (overrides config)")
    ap.add_argument("--no-video", action="store_true", help="skip the annotated video (faster)")
    ap.add_argument("--no-clips", action="store_true", help="skip the driver-yielding video clips")
    ap.add_argument("--progress-every", type=float, default=30.0,
                    help="seconds between status lines (default 30)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cal = load_calibration(args.calibration)
    det = cfg["detection"]
    stride = args.stride or int(det["stride"])
    tz = ZoneInfo(cfg["timezone"])
    start_sec = args.start_min * 60
    duration = args.minutes * 60 if args.minutes else None
    partial = bool(start_sec or duration)
    if args.out:
        out = args.out
    elif partial:
        end = f"{args.start_min + args.minutes:g}" if args.minutes else "end"
        out = Path("out") / f"{args.video.stem}_{args.start_min:g}-{end}min"
    else:
        out = Path("out") / args.video.stem
    out.mkdir(parents=True, exist_ok=True)

    # 1. Read the start time from the original's metadata, then strip audio.
    #    Nothing after this line touches the original file.
    if args.start_time:
        start_time = datetime.fromisoformat(args.start_time).replace(tzinfo=tz)
        start_source = "--start-time override"
    else:
        start_time = recording_start(args.video, cfg["timezone"])
        start_source = "video metadata (creation date)"
        if start_time is None:
            sys.exit("No creation date in the video metadata; pass --start-time.")
    status(f"Copying video without audio to {args.work}/ (skipped if already done)...")
    video = strip_audio(args.video, args.work / f"{args.video.stem}.noaudio.mov")

    reader = FrameReader(video, start=start_sec, duration=duration)
    info = reader.info
    expected = min(duration or info.duration, info.duration - start_sec)
    if expected <= 0:
        sys.exit(f"--start-min {args.start_min:g} is past the end of the video "
                 f"({info.duration / 60:.1f} min)")
    detector = Detector(cfg, info.fps, stride)
    status(f"Model {Path(det['model']).name} on {detector.device}, detecting every {stride} frame(s)")

    ov = cfg["output"]["overlays"]
    limit = float(cfg["speed_limit_mph"])
    trap_measured = bool(cal["speed_trap"].get("distance_measured"))

    analyzer = new_analyzer(cfg, cal, start_time, out)
    writer = None
    if cfg["output"]["video"] and not args.no_video:
        writer = VideoWriter(out / "annotated.mp4", info.width, info.height, info.fps,
                             cfg["output"]["height"], cfg["output"]["bitrate"])
    scene = render.SceneOverlay(cal, (info.height, info.width), zones=ov["zones"], gates=ov["gates"])

    clip_start = start_time + timedelta(seconds=start_sec)
    status(f"Processing {expected / 60:.1f} min of video starting {clip_start:%a %b %d %H:%M:%S} PT. "
           f"Status every {args.progress_every:g}s. Outputs -> {out}/")

    current: list[tuple] = []  # (box, label, color) from the latest detection frame
    t0 = time.time()
    next_status = t0 + args.progress_every
    first_pts = last_pts = None
    n = 0
    for n, (pts, frame) in enumerate(reader):
        if first_pts is None:
            first_pts = pts
        last_pts = pts
        if n % stride == 0:
            tracked = detector.step(pts, frame, analyzer)

            current = []
            for box, tid in zip(tracked["vehicle"].xyxy, tracked["vehicle"].tracker_id):
                tr = analyzer.vehicles.get(("vehicle", int(tid)))
                if tr is None:
                    continue
                if tr.movement in ("left_turn", "right_turn", "u_turn"):
                    color, label = render.GRAY, f"#{tr.tid} TURN"
                elif tr.speed_mph is not None:
                    color = render.speed_color(tr.speed_mph, limit)
                    label = f"#{tr.tid} {tr.speed_mph:.0f} mph" + ("" if trap_measured else " est")
                else:
                    color, label = render.BLUE, f"#{tr.tid} {tr.cls}"
                current.append((box, label, color, tr))
            for box, tid in zip(tracked["bicycle"].xyxy, tracked["bicycle"].tracker_id):
                current.append((box, f"bike #{tid}", render.CYAN, None))
            for box, tid in zip(tracked["person"].xyxy, tracked["person"].tracker_id):
                tr = analyzer.people.get(int(tid))
                label = "crossing" if tr and tr.crossed_pts is not None else None
                current.append((box, label, render.MAGENTA, None))

        if writer:
            scene.apply(frame)
            for box, label, color, tr in current:
                if ov.get("trails") and tr is not None and len(tr.path) > 1:
                    p = np.asarray([(x, y) for _, x, y in tr.path[-30:]], np.int32)
                    cv2.polylines(frame, [p], False, color, 2, cv2.LINE_AA)
                render.draw_box(frame, box, color, label)
            if ov["timestamp"]:
                ts = start_time + timedelta(seconds=pts)
                render.draw_timestamp(frame, ts.strftime("%a %b %d %Y %H:%M:%S PT"))
            if ov["counters"]:
                c = analyzer.counts
                labels = list(cfg["direction_labels"].values())
                lines = [f"Vehicles: {c['vehicle_total']}  ("
                         + " / ".join(f"{l[:1].upper()}B {c[f'vehicle_{l}']}" for l in labels) + ")",
                         f"Pedestrians crossing: {c['ped_total']}",
                         f"Bicycles: {c['bicycle_total']}",
                         f"Over {limit:g} mph: {c['over_limit']} of {c['speed_measured']}"
                         + ("" if trap_measured else " (est.)")]
                render.draw_panel(frame, lines)
            writer.write(frame)

        now = time.time()
        done = pts - (first_pts or 0)
        if now >= next_status and done > 0:
            next_status = now + args.progress_every
            eta = (expected - done) / done * (now - t0) if done > 0 else 0
            c = analyzer.counts
            status(f"{done / 60:5.1f}/{expected / 60:.1f} min of video ({100 * done / expected:3.0f}%) | "
                   f"video time {start_time + timedelta(seconds=pts):%H:%M:%S} | "
                   f"{(n + 1) / (now - t0):4.1f} fps | "
                   f"ETA {eta / 60:4.1f} min (~{datetime.now() + timedelta(seconds=eta):%H:%M}) | "
                   f"vehicles {c['vehicle_total']}, peds {c['ped_total']}, bikes {c['bicycle_total']}")

    elapsed = time.time() - t0
    status(f"Processed {n + 1} frames in {elapsed / 60:.1f} min. Finishing video and writing reports...")
    analyzer.finalize_stale(last_pts or 0, force=True)
    if writer:
        writer.close()

    write_run(out, analyzer, cfg, cal, args.calibration, source=args.video.name,
              start_source=start_source, video=video, fps=info.fps, width=info.width,
              height=info.height, first_pts=first_pts or 0, last_pts=last_pts or 0,
              detector=detector, stride=stride, frames=n + 1, minutes=elapsed / 60)
    if cfg["yielding"]["enabled"]:
        status("Checking whether drivers yielded to pedestrians...")
        analyze_yields(out, cfg, cal, make_clips=not args.no_clips, log=status)
    s = write_summary(out, cfg)
    status(f"Done. Counted {s['vehicles']['counted_total']} vehicles and "
           f"{s['pedestrians']['crossings']} pedestrian crossings. See {out}/summary.md")


if __name__ == "__main__":
    main()
