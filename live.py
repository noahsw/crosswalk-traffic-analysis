"""Watch a live camera (e.g. a Tapo C120) and save a clip of the drivers who don't yield to each
pedestrian, and of every pedestrian crossing.

    uv run python live.py                         # the camera at live.host in config.yaml
    uv run python live.py --host 192.168.1.50
    uv run python live.py --hours 8               # stop by itself after 8 hours
    uv run python live.py --show                  # also show the tracking in a window
    uv run python live.py input/VIDEO.MOV --calibration calibration.yaml   # rehearse on a recording

The camera account comes from the macOS Keychain (see the README), or from the TAPO_USER and
TAPO_PASSWORD environment variables if both are set.
Press Ctrl+C to stop. Clips of drivers not yielding (one per pedestrian, with every driver who
didn't yield to them) are saved to out/did_not_yield/ (one folder for every session), and of
each crossing to out/live_<date>_<time>/crossing_clips/, as they happen (about 20 s after each crossing). Each crossing's video is also kept as recorded, with no
boxes or captions, in raw_crossings/ beside it, to replay later (tests/test_replay.py).
Each crossing and judged driver is added to yield_log.csv right away.
On stop, the usual CSVs and summary.md are written for the whole session.

While it runs, the Mac is kept from sleeping on its own (closing the lid still sleeps it). After
a sleep or a camera outage it reconnects and carries on; the time with no video is listed in
summary.md and left out of the per-hour rates.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from itertools import chain
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2

from traffic.clips import by_pedestrian, caption_for, crossing_caption, live_clip_dir
from traffic.config import load_calibration, load_config
from traffic.live import ClipSaver, LiveSource, LiveYieldWatcher, camera_url, probe_stream, redact
from traffic.pipeline import Annotator, Detector, new_analyzer, write_run
from traffic.report import write_summary
from traffic.video import VideoError
from yielding import analyze_yields


GAP_SECONDS = 2.0   # a longer jump between frames means the video stopped (asleep, reconnect)
WINDOW = "live.py - press q in this window to hide it, Ctrl+C in the terminal to stop"
WINDOW_WIDTH = 1280


def status(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class Window:
    """Shows each frame with the tracking overlay. Closing it (or q) only hides it."""

    def __init__(self, annotator: Annotator):
        self.annotator = annotator
        self.open = True
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    def show(self, frame, pts: float, analyzer) -> None:
        if not self.open:
            return
        self.annotator.draw(frame, pts, analyzer)
        h, w = frame.shape[:2]
        cv2.imshow(WINDOW, cv2.resize(frame, (WINDOW_WIDTH, round(h * WINDOW_WIDTH / w)),
                                      interpolation=cv2.INTER_AREA))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            self.close()
            status("Window closed; still watching. Press Ctrl+C to stop.")

    def close(self) -> None:
        if self.open:
            self.open = False
            cv2.destroyAllWindows()
            cv2.waitKey(1)


def stop_on_sigterm() -> None:
    """Treat `kill` (SIGTERM) like Ctrl+C, so the session still finishes and writes its report."""
    def handler(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, signal.default_int_handler)   # even if started in the background


def keep_awake() -> None:
    """Stop the Mac from sleeping while this process runs (idle sleep, and any sleep on power)."""
    if shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?",
                    help="rtsp:// URL, or a video file to play back in real time as if it were live "
                         "(default: the camera in config.yaml)")
    ap.add_argument("--host", help="camera IP address (overrides live.host)")
    ap.add_argument("--stream", help="stream1 = full resolution, stream2 = low resolution (overrides live.stream)")
    ap.add_argument("--calibration", type=Path, default=None,
                    help="scene geometry drawn on this camera's view (default: live.calibration)")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--out", type=Path, default=None, help="output folder (default out/live_<date>_<time>)")
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--stride", type=int, default=None, help="detect every Nth frame (overrides live.stride)")
    ap.add_argument("--hours", type=float, default=None, help="stop after this many hours")
    ap.add_argument("--no-clips", action="store_true", help="judge drivers but don't save clips")
    ap.add_argument("--show", action="store_true",
                    help="show the camera with the tracking overlay in a window")
    ap.add_argument("--progress-every", type=float, default=60.0,
                    help="seconds between status lines (default 60)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    live = cfg["live"]
    cal_path = args.calibration or Path(live["calibration"])
    if not cal_path.exists():
        sys.exit(f"{cal_path} not found. Draw the scene on the camera's view first: "
                 f"uv run python calibrate.py --camera")
    cal = load_calibration(cal_path)
    tz = ZoneInfo(cfg["timezone"])
    try:
        src = args.source or camera_url(live, args.host, args.stream)
        status(f"Connecting to {redact(src)}...")
        width, height, fps = probe_stream(src)
    except VideoError as ex:
        sys.exit(str(ex))
    size = cal.get("image_size")
    if size and list(size) != [width, height]:
        sys.exit(f"The stream is {width}x{height}, but {cal_path} was drawn on a {size[0]}x{size[1]} "
                 f"view. Redraw it on the camera: uv run python calibrate.py --camera")
    stride = args.stride or int(live["stride"])
    out = args.out or Path("out") / f"live_{datetime.now(tz):%Y-%m-%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    keep_awake()
    stop_on_sigterm()
    # Load the model before connecting, so the stream doesn't back up while it loads.
    detector = Detector(cfg, fps, stride, cal)
    status(f"{width}x{height} at {fps:g} fps. Model {Path(cfg['detection']['model']).name} on "
           f"{detector.device}, detecting every {stride} frame(s)")
    work = args.work / "live" / out.name
    source = LiveSource(src, work, width, height, fps, tz,
                        segment_seconds=float(live["segment_seconds"]),
                        keep_seconds=float(live["keep_minutes"]) * 60,
                        buffer_seconds=float(live["max_lag_seconds"]), log=status)
    frames = iter(source)
    first = next(frames, None)
    if first is None:
        source.stop()
        sys.exit("No video arrived from the camera.")
    start_time = source.start_time
    analyzer = new_analyzer(cfg, cal, start_time, out)

    y = cfg["yielding"]
    watcher = LiveYieldWatcher(cfg, cal, analyzer) if y["enabled"] else None
    wanted = set(y.get("clip_outcomes") or ["did_not_yield"])
    every_crossing = bool(y["clip_every_crossing"])
    saver = None
    if watcher and not args.no_clips:
        saver = ClipSaver(source, cal, start_time, float(y["clip_seconds_before"]),
                          float(y["clip_seconds_after"]),
                          crossing_seconds_before=float(y["crossing_clip_seconds_before"]), log=status)
    outcomes: Counter = Counter()
    log_path = out / "yield_log.csv"   # written as it goes, in case the session never finishes
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["time_local", "outcome", "direction", "vehicle_id", "clip", "what"])

    def log_row(when, outcome, direction, vehicle_id, clip, what) -> None:
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([when.isoformat(timespec="seconds"), outcome, direction,
                                    vehicle_id, clip.name if clip else "", what])

    def handle(judged) -> None:
        clips = {}   # id(event) -> its clip: one per pedestrian and outcome, however many drivers
        if saver:
            vehicles = list({id(v): v for j in judged for v in j.vehicles}.values())
            for outcome in wanted:
                for group in by_pedestrian([e for j in judged for e in j.events if e.outcome == outcome]):
                    path = saver.add(group, vehicles, live_clip_dir(out, outcome))
                    clips.update({id(e): path for e in group})
        for j in judged:
            c = j.crossing
            when = start_time + timedelta(seconds=c.wait_start)
            status(f"{when:%H:%M:%S} {crossing_caption([c])}")
            if saver and every_crossing:
                saver.add_crossing(j, out / "crossing_clips", raw_dir=out / "raw_crossings")
            log_row(when, "pedestrian_crossing", c.direction, "", None, crossing_caption([c]))
            for e in j.events:
                outcomes[e.outcome] += 1
                when = start_time + timedelta(seconds=e.passes)
                status(f"{when:%H:%M:%S} {caption_for(e)}")
                log_row(when, e.outcome, e.direction or "", e.vehicle.tid, clips.get(id(e)),
                        caption_for(e))

    window = Window(Annotator(cfg, cal, start_time, (height, width))) if args.show else None
    status(f"Watching since {start_time:%a %b %d %H:%M:%S}. Press Ctrl+C to stop. Outputs -> {out}/")
    t0 = time.time()
    next_status = t0 + args.progress_every
    since_status, frames_since = t0, 0
    dropped_before = 0        # source.dropped at the last status line
    gap_since_status = True   # also skip the first interval, while the model warms up
    next_poll = 0.0
    warned_lag = False
    first_pts = last_pts = first[0]
    gaps: list[tuple[float, float]] = []
    n = 0
    try:
        for n, (pts, frame) in enumerate(chain([first], frames)):
            if pts - last_pts > GAP_SECONDS:
                # Close everything seen before the gap, so it's judged now and its clips can be
                # cut, and nothing after the gap is joined to it.
                gaps.append((last_pts, pts))
                gap_since_status = True   # a reconnect delivers a burst; skipping that is fine
                status(f"No video from {start_time + timedelta(seconds=last_pts):%H:%M:%S} to "
                       f"{start_time + timedelta(seconds=pts):%H:%M:%S} "
                       f"({(pts - last_pts) / 60:.1f} min; asleep or camera offline). Carrying on.")
                analyzer.finalize_stale(last_pts, force=True)
                detector.forget()
                if watcher:
                    handle(watcher.poll(pts))
            last_pts = pts
            frames_since += 1
            if n % stride == 0:
                tracked = detector.step(pts, frame, analyzer)
                if window:
                    window.annotator.update(tracked, analyzer)
            # Skip drawing while behind, so the window never costs detection frames.
            if window and source.lag_seconds < 0.5:
                window.show(frame, pts, analyzer)
            if watcher and pts >= next_poll:
                handle(watcher.poll(pts))
                next_poll = pts + 1.0
            now = time.time()
            if now >= next_status:
                if source.dropped > dropped_before and not gap_since_status and not warned_lag:
                    warned_lag = True
                    status("Detection can't keep up with the camera, so some frames are being skipped. "
                           f"If this keeps happening, use --stride {stride + 1} or a smaller model "
                           "(detection.model).")
                dropped_before, gap_since_status = source.dropped, False
                rate = frames_since / max(now - since_status, 1e-6)
                next_status, since_status, frames_since = now + args.progress_every, now, 0
                c = analyzer.counts
                status(f"{start_time + timedelta(seconds=pts):%H:%M:%S} | {rate:4.1f} fps | "
                       f"skipped {source.dropped} frame(s) | vehicles {c['vehicle_total']}, "
                       f"peds {c['ped_total']} | did not yield {outcomes['did_not_yield']}, "
                       f"yielded {outcomes['yielded']} | clips saved {saver.saved if saver else 0}")
            if args.hours and pts - first_pts >= args.hours * 3600:
                break
    except KeyboardInterrupt:
        print()
    status("Stopping...")
    if window:
        window.close()
    source.stop()
    analyzer.finalize_stale(last_pts, force=True)
    if watcher:
        handle(watcher.poll(last_pts, force=True))
    if saver:
        status("Saving any remaining clips...")
        saver.close()
    elapsed = time.time() - t0

    camera = f"live camera ({args.stream or live['stream']})"
    write_run(out, analyzer, cfg, cal, cal_path,
              source=camera if not args.source or args.source.startswith("rtsp://")
              else f"{Path(args.source).name} (played back live)",
              start_source="computer clock (live stream)", video=None, fps=fps, width=width,
              height=height, first_pts=first_pts, last_pts=last_pts, detector=detector,
              stride=stride, frames=n + 1, minutes=elapsed / 60, gaps=gaps)
    if y["enabled"]:
        analyze_yields(out, cfg, cal, make_clips=False, log=status)
    s = write_summary(out, cfg)
    shutil.rmtree(work, ignore_errors=True)   # the rolling recording; the clips are already saved
    status(f"Done. Counted {s['vehicles']['counted_total']} vehicles and "
           f"{s['pedestrians']['crossings']} pedestrian crossings; saved "
           f"{saver.saved if saver else 0} clip(s). See {out}/summary.md")


if __name__ == "__main__":
    main()
