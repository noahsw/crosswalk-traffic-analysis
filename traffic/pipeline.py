"""Detection, tracking and run outputs shared by analyze.py (recorded video) and live.py (camera)."""

from __future__ import annotations

import json
import os
import pickle
import shutil
import warnings
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from . import render
from .report import ped_review_rows, ped_rows, vehicle_rows
from .tracks import Analyzer

# Known: supervision's ByteTrack is deprecated (pinned <0.31, see TODO.md).
warnings.filterwarnings("ignore", message="The `ByteTrack` was deprecated")

KINDS = ("vehicle", "bicycle", "person")


def pick_device(pref: str) -> str:
    import torch
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(path: str):
    from ultralytics import YOLO
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        # Ultralytics downloads known model names into the working directory.
        cwd = os.getcwd()
        os.chdir(p.parent)
        try:
            YOLO(p.name)
        finally:
            os.chdir(cwd)
    return YOLO(str(p))


class Detector:
    """YOLO detection and one ByteTrack tracker per kind of object, feeding an Analyzer."""

    def __init__(self, cfg: dict, fps: float, stride: int):
        det = self.det = cfg["detection"]
        self.model = load_model(det["model"])
        self.device = pick_device(det["device"])
        self.names = self.model.names
        want = set(det["vehicle_classes"]) | set(det["bicycle_classes"]) | set(det["person_classes"])
        self.class_ids = [i for i, n in self.names.items() if n in want]
        det_rate = fps / stride
        lost = max(1, int(round(float(cfg["tracking"]["lost_track_seconds"]) * det_rate)))
        self.trackers = {k: sv.ByteTrack(frame_rate=int(round(det_rate)), lost_track_buffer=lost)
                         for k in KINDS}
        self.kind_of = {}
        for i, n in self.names.items():
            for k in KINDS:
                if n in det[f"{k}_classes"]:
                    self.kind_of[i] = k

    def step(self, pts: float, frame, analyzer: Analyzer) -> dict:
        """Detect, track and update `analyzer` for one frame. Returns the tracked detections by kind."""
        det = self.det
        r = self.model.predict(frame, imgsz=det["imgsz"], conf=det["confidence"], device=self.device,
                               classes=self.class_ids, agnostic_nms=True, verbose=False)[0]
        d = sv.Detections.from_ultralytics(r)
        kinds = np.array([self.kind_of[c] for c in d.class_id]) if len(d) else np.array([])
        tracked = {}
        for k, tracker in self.trackers.items():
            tracked[k] = tracker.update_with_detections(d[kinds == k] if len(d) else d)
        names = self.names
        for k in ("vehicle", "bicycle"):
            t = tracked[k]
            analyzer.update_vehicles(pts, frame, t.xyxy, t.tracker_id, [names[c] for c in t.class_id], k)
        riders = np.concatenate([tracked["bicycle"].xyxy,
                                 tracked["vehicle"].xyxy[[names[c] == "motorcycle" for c in tracked["vehicle"].class_id]]])
        analyzer.update_people(pts, frame, tracked["person"].xyxy, tracked["person"].tracker_id,
                               tracked["vehicle"].xyxy, riders)
        analyzer.finalize_stale(pts)
        return tracked

    def forget(self) -> None:
        """Drop every track, so objects after a gap in the video aren't joined to ones before it.
        Track ids keep counting up, so they stay unique."""
        for tracker in self.trackers.values():
            for _ in range(tracker.max_time_lost + 2):
                tracker.update_with_detections(sv.Detections.empty())


def new_analyzer(cfg: dict, cal: dict, start_time, out: Path) -> Analyzer:
    """An Analyzer that saves movement_check/ and ped_check/ review images into `out`."""
    for sub in ("movement_check", "ped_check"):
        shutil.rmtree(out / sub, ignore_errors=True)   # don't mix images from earlier runs
        (out / sub).mkdir(parents=True)

    def save_snapshot(tr):
        img = tr.snapshot
        pts = np.asarray([(x, y) for _, x, y in tr.path], np.int32)
        cv2.polylines(img, [pts], False, render.YELLOW, 3, cv2.LINE_AA)
        render.draw_box(img, tr.snapshot_box, render.YELLOW, f"#{tr.tid}")
        render.put_label(img, f"{tr.movement}: entry={tr.entry} exit={tr.exit}"
                         + (" (inferred)" if tr.inferred else ""), (12, 12), render.WHITE,
                         anchor="tl", scale=0.9)
        cv2.imwrite(str(out / "movement_check" / f"{tr.movement}_{tr.tid}.jpg"), img)

    def save_ped_snapshot(tr, reason):
        img = tr.snapshot
        cv2.polylines(img, [np.asarray(cal["crosswalk"], np.int32)], True, render.CROSSWALK_COLOR, 2)
        if len(tr.path) > 1:
            cv2.polylines(img, [np.asarray([(x, y) for _, x, y in tr.path], np.int32)], False,
                          render.YELLOW, 3, cv2.LINE_AA)
        render.draw_box(img, tr.snapshot_box, render.MAGENTA, f"P{tr.tid}")
        ts = start_time + timedelta(seconds=tr.first_pts)
        render.put_label(img, f"{reason.replace('_', ' ')}  |  first seen {ts:%H:%M:%S}  |  "
                         f"video {tr.first_pts / 60:.0f}:{tr.first_pts % 60:04.1f}",
                         (12, 12), render.WHITE, anchor="tl", scale=0.9)
        cv2.imwrite(str(out / "ped_check" / f"{ts:%H%M%S}_{reason}_P{tr.tid}.jpg"), img)

    return Analyzer(cfg, cal, start_time, on_snapshot_needed=save_snapshot,
                    on_ped_snapshot=save_ped_snapshot)


def write_run(out: Path, analyzer: Analyzer, cfg: dict, cal: dict, calibration_path: Path, *,
              source: str, start_source: str, video: Path | None, fps: float, width: int, height: int,
              first_pts: float, last_pts: float, detector: Detector, stride: int, frames: int,
              minutes: float, gaps: list[tuple[float, float]] = ()) -> None:
    """Write the CSVs, tracks.pkl and run_meta.json for a finished run.

    `video` is the audio-free copy the clips are cut from, or None for a live session.
    `gaps` are (start, end) times when a live session saw no video (asleep, camera offline).
    """
    start_time = analyzer.start_time
    vehicle_rows(analyzer).to_csv(out / "vehicles.csv", index=False)
    ped_rows(analyzer).to_csv(out / "pedestrians.csv", index=False)
    ped_review_rows(analyzer).to_csv(out / "pedestrians_reviewed.csv", index=False)
    # Keep every track, so later analysis (e.g. driver yielding) doesn't need to re-run detection.
    with open(out / "tracks.pkl", "wb") as f:
        pickle.dump({"vehicles": analyzer.vehicle_records, "pedestrians": analyzer.ped_records,
                     "start_time": start_time, "video": str(video) if video else None,
                     "source": source, "fps": fps, "width": width, "height": height}, f)
    trap = cal["speed_trap"]
    det = cfg["detection"]
    meta = {
        "source": source,
        "window_start": (start_time + timedelta(seconds=first_pts)).isoformat(),
        "window_end": (start_time + timedelta(seconds=last_pts)).isoformat(),
        "start_time_source": start_source,
        "not_watched": [{"start": (start_time + timedelta(seconds=a)).isoformat(),
                         "end": (start_time + timedelta(seconds=b)).isoformat(),
                         "minutes": round((b - a) / 60, 1)} for a, b in gaps],
        "speed_limit_mph": float(cfg["speed_limit_mph"]),
        "direction_labels": list(cfg["direction_labels"].values()),
        "calibration": {"file": str(calibration_path), "created": cal.get("created"),
                        "distance_ft": trap["distance_ft"],
                        "distance_measured": bool(trap.get("distance_measured")),
                        "distance_notes": trap.get("distance_notes", ""),
                        "count_line_leg": cal.get("count_line_leg")},
        "gate_labels": cal.get("gate_labels", {}),
        "site_name": cal.get("site_name"),
        "processing": {"model": Path(det["model"]).name, "imgsz": det["imgsz"],
                       "confidence": det["confidence"], "stride": stride, "device": detector.device,
                       "frames": frames, "minutes": round(minutes, 1),
                       "ignored_tracks": analyzer.ignored_tracks},
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
