"""Interactive calibration: draw the count line, speed trap, gates and crosswalk.

    uv run python calibrate.py input/VIDEO.MOV
    uv run python calibrate.py input/VIDEO.MOV --preview-only
    uv run python calibrate.py --camera          # a frame from the live camera (see live.py)

Each step shows the current shape (if any) from calibration.yaml.
  click      add a point
  Enter      accept the new points (or keep the existing shape if none clicked)
  u          undo last point
  c          clear the points clicked in this step
  q / Esc    quit without saving
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import yaml

from traffic import render
from traffic.config import load_config, save_calibration
from traffic.live import camera_url, grab_frame
from traffic.video import VideoError, read_frame_at, strip_audio

STEPS = [
    # (key path, points, instructions)
    (("count_line",), 2, "COUNT LINE: 2 points across both lanes. Every vehicle crossing it is counted."),
    (("speed_trap", "line_a"), 2, "SPEED TRAP line A: start at a measurable mark on the far curb, go across the road."),
    (("speed_trap", "line_b"), 2, "SPEED TRAP line B: the other measurable mark on the far curb, across the road."),
    (("crosswalk",), 4, "CROSSWALK: 4 corners. Near-side 2 corners first, then far-side 2 corners."),
]
WINDOW = "calibrate"


def get(cal: dict, path: tuple):
    for k in path:
        cal = (cal or {}).get(k)
    return cal


def put(cal: dict, path: tuple, value) -> None:
    for k in path[:-1]:
        cal = cal.setdefault(k, {})
    cal[path[-1]] = value


def click_step(frame, cal: dict, path: tuple, n: int, text: str):
    existing = get(cal, path)
    pts: list[list[int]] = []
    scale = frame.shape[1] / 1920

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append([int(x), int(y)])

    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        img = frame.copy()
        if existing and not pts:
            cv2.polylines(img, [np.asarray(existing, np.int32)], n > 2, render.GRAY, 2, cv2.LINE_AA)
        if pts:
            cv2.polylines(img, [np.asarray(pts, np.int32)], n > 2 and len(pts) == n,
                          render.YELLOW, 2, cv2.LINE_AA)
            for i, p in enumerate(pts):
                cv2.circle(img, tuple(p), 5, render.RED, -1)
                cv2.putText(img, str(i + 1), (p[0] + 6, p[1] - 6), render.FONT, 0.6, render.RED, 2)
        render.put_label(img, text, (10, 10), render.WHITE, scale=0.7 * scale, anchor="tl")
        hint = (f"{len(pts)}/{n} points.  Enter=accept  u=undo  c=clear  q=quit"
                + ("  (Enter with no clicks keeps the gray shape)" if existing else ""))
        render.put_label(img, hint, (10, 50), render.WHITE, scale=0.55 * scale, anchor="tl")
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):
            if len(pts) == n:
                return pts
            if not pts and existing:
                return existing
        elif key == ord("u") and pts:
            pts.pop()
        elif key == ord("c"):
            pts.clear()
        elif key in (ord("q"), 27):
            raise SystemExit("Quit without saving.")


def ask(prompt: str, default):
    raw = input(f"{prompt} [{default}]: ").strip()
    return type(default)(raw) if raw else default


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", help="video file, or an rtsp:// camera URL")
    ap.add_argument("--camera", action="store_true",
                    help="use a frame from the live camera at live.host in config.yaml")
    ap.add_argument("--host", help="camera IP address (overrides live.host)")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--calibration", type=Path, default=None,
                    help="default calibration.yaml, or live.calibration in config.yaml for a camera")
    ap.add_argument("--time", type=float, default=None, help="seconds into the video for the reference frame")
    ap.add_argument("--gates", default=None,
                    help="comma-separated compass names of the intersection legs "
                         "(default: the gates already in the calibration file, else west,east,north,south)")
    ap.add_argument("--preview-only", action="store_true", help="just redraw calibration_preview.jpg")
    ap.add_argument("--work", type=Path, default=Path("work"))
    args = ap.parse_args()

    live = args.camera or bool(args.video and args.video.startswith("rtsp://"))
    if not live and not args.video:
        ap.error("give a video file, or --camera")
    if args.calibration is None:
        args.calibration = Path(load_config(args.config)["live"]["calibration"] if live else "calibration.yaml")
    cal = yaml.safe_load(args.calibration.read_text()) if args.calibration.exists() else {}
    t = args.time if args.time is not None else cal.get("reference_frame_seconds", 5)
    if live:
        try:
            frame = grab_frame(args.video or camera_url(load_config(args.config)["live"], args.host))
        except VideoError as ex:
            raise SystemExit(str(ex))
    else:
        video = Path(args.video)
        frame = read_frame_at(strip_audio(video, args.work / f"{video.stem}.noaudio.mov"), t)
    preview_path = args.calibration.with_name(f"{args.calibration.stem}_preview.jpg")

    if not args.preview_only:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1600, int(1600 * frame.shape[0] / frame.shape[1]))
        steps = list(STEPS)
        gate_names = ([g.strip() for g in args.gates.split(",") if g.strip()] if args.gates
                      else list(cal.get("gates") or {}) or ["west", "east", "north", "south"])
        for g in gate_names:
            steps.insert(-1, (("gates", g), 2,
                              f"GATE '{g}': 2 points across the {g} leg, where vehicles enter/leave the scene."))
        for path, n, text in steps:
            put(cal, path, click_step(frame, cal, path, n, text))
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        trap = cal.setdefault("speed_trap", {})
        print("\nSpeed trap distance: the real-world distance between where line A and line B")
        print("meet the far curb, measured along the curb with a tape.")
        trap["distance_ft"] = ask("Distance in feet", float(trap.get("distance_ft", 35.0)))
        trap["distance_measured"] = ask("Was that tape-measured? (y/n)",
                                        "y" if trap.get("distance_measured") else "n").lower().startswith("y")
        trap["distance_notes"] = ask("Notes (what was measured)", trap.get("distance_notes", ""))
        cal["count_line_leg"] = ask("Which leg is the count line on?", cal.get("count_line_leg", "east"))
        cal["site_name"] = ask("Site name for report titles", cal.get("site_name") or "crosswalk")
        cal["created"] = date.today().isoformat()
        cal["image_size"] = [frame.shape[1], frame.shape[0]]
        if not live:
            cal["reference_frame_seconds"] = t
        save_calibration(cal, args.calibration)
        print(f"Saved {args.calibration}")

    cv2.imwrite(str(preview_path), render.draw_calibration_preview(frame, cal))
    print(f"Wrote {preview_path} - open it and check every line is where you expect.")


if __name__ == "__main__":
    main()
