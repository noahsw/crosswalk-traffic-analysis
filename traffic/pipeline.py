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

from . import coreml, render
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


def load_model(path: str, imgsz: int = 960):
    """A YOLO model: a .pt file (downloaded if missing), or a CoreML .mlpackage (exported if
    missing, `imgsz` wide)."""
    from ultralytics import YOLO
    p = Path(path)
    if p.suffix == ".mlpackage":
        if not p.exists():
            print(f"Exporting {p} for the Neural Engine (once, about a minute)...", flush=True)
            coreml.export(p, imgsz)
        return YOLO(str(p), task="detect")
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


def crosswalk_zoom(cal: dict, crop_w: int, crop_h: int, pad: float) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) of a crop around the crosswalk and the waiting areas at its ends, the
    model's input size (larger if the crosswalk doesn't fit), or None if that's the whole frame."""
    w, h = cal.get("image_size") or (1920, 1080)
    cw = np.asarray(cal["crosswalk"], float)
    (x0, y0), (x1, y1) = cw.min(0) - pad, cw.max(0) + pad
    scale = max(1.0, (x1 - x0) / crop_w, (y1 - y0) / crop_h)
    cw_, ch_ = round(crop_w * scale), round(crop_h * scale)
    if cw_ >= w or ch_ >= h:
        return None
    left = int(min(max(0, (x0 + x1 - cw_) / 2), w - cw_))
    top = int(min(max(0, (y0 + y1 - ch_) / 2), h - ch_))
    return left, top, left + cw_, top + ch_


class Detector:
    """YOLO detection and one ByteTrack tracker per kind of object, feeding an Analyzer."""

    def __init__(self, cfg: dict, fps: float, stride: int, cal: dict | None = None):
        det = self.det = cfg["detection"]
        self.model = load_model(det["model"], int(det["imgsz"]))
        if Path(det["model"]).suffix == ".mlpackage":
            # CoreML picks the Neural Engine itself, and its input size is fixed at export.
            self.device, self.predict_device = "coreml", "cpu"
            self.imgsz = coreml.input_size(det["model"])
        else:
            self.device = self.predict_device = pick_device(det["device"])
            self.imgsz = det["imgsz"]
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
        self.person_ids = [i for i, k in self.kind_of.items() if k == "person"]
        self.zoom = None
        if cal is not None and det.get("crosswalk_zoom"):
            h, w = self.imgsz if isinstance(self.imgsz, tuple) else coreml.shape_for(self.imgsz)
            self.zoom = crosswalk_zoom(cal, w, h, float(cfg["yielding"]["waiting_area_px"]))

    def step(self, pts: float, frame, analyzer: Analyzer) -> dict:
        """Detect, track and update `analyzer` for one frame. Returns the tracked detections by kind."""
        det = self.det
        r = self.model.predict(frame, imgsz=self.imgsz, conf=det["confidence"], device=self.predict_device,
                               classes=self.class_ids, agnostic_nms=True, verbose=False)[0]
        d = sv.Detections.from_ultralytics(r)
        if self.zoom:
            x0, y0, x1, y1 = self.zoom
            r = self.model.predict(frame[y0:y1, x0:x1], imgsz=self.imgsz, conf=det["confidence"],
                                   device=self.predict_device, classes=self.person_ids, verbose=False)[0]
            z = sv.Detections.from_ultralytics(r)
            if len(z):
                z.xyxy += np.array([x0, y0, x0, y0], dtype=z.xyxy.dtype)
                d = sv.Detections.merge([d, z]).with_nms(threshold=0.5, class_agnostic=True)
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


class Annotator:
    """The tracking overlay (zones, boxes, timestamp, counters) drawn on frames, as in annotated.mp4."""

    def __init__(self, cfg: dict, cal: dict, start_time, shape):
        self.cfg, self.start_time = cfg, start_time
        self.ov = ov = cfg["output"]["overlays"]
        self.limit = float(cfg["speed_limit_mph"])
        self.trap_measured = bool(cal["speed_trap"].get("distance_measured"))
        self.scene = render.SceneOverlay(cal, shape, zones=ov["zones"], gates=ov["gates"])
        self.current: list[tuple] = []  # (box, label, color, track) from the latest detection frame

    def update(self, tracked: dict, analyzer: Analyzer) -> None:
        """Take the boxes to draw from a detection frame (Detector.step's result)."""
        current = []
        for box, tid in zip(tracked["vehicle"].xyxy, tracked["vehicle"].tracker_id):
            tr = analyzer.vehicles.get(("vehicle", int(tid)))
            if tr is None:
                continue
            if tr.movement in ("left_turn", "right_turn", "u_turn"):
                color, label = render.GRAY, f"#{tr.tid} TURN"
            elif tr.speed_mph is not None:
                color = render.speed_color(tr.speed_mph, self.limit)
                label = f"#{tr.tid} {tr.speed_mph:.0f} mph" + ("" if self.trap_measured else " est")
            else:
                color, label = render.BLUE, f"#{tr.tid} {tr.cls}"
            current.append((box, label, color, tr))
        for box, tid in zip(tracked["bicycle"].xyxy, tracked["bicycle"].tracker_id):
            current.append((box, f"bike #{tid}", render.CYAN, None))
        for box, tid in zip(tracked["person"].xyxy, tracked["person"].tracker_id):
            tr = analyzer.people.get(int(tid))
            label = "crossing" if tr and tr.crossed_pts is not None else None
            current.append((box, label, render.MAGENTA, None))
        self.current = current

    def draw(self, frame, pts: float, analyzer: Analyzer) -> None:
        """Draw the overlay on `frame` in place."""
        ov = self.ov
        self.scene.apply(frame)
        for box, label, color, tr in self.current:
            if ov.get("trails") and tr is not None and len(tr.path) > 1:
                p = np.asarray([(x, y) for _, x, y in tr.path[-30:]], np.int32)
                cv2.polylines(frame, [p], False, color, 2, cv2.LINE_AA)
            render.draw_box(frame, box, color, label)
        if ov["timestamp"]:
            ts = self.start_time + timedelta(seconds=pts)
            render.draw_timestamp(frame, ts.strftime("%a %b %d %Y %H:%M:%S PT"))
        if ov["counters"]:
            c = analyzer.counts
            labels = list(self.cfg["direction_labels"].values())
            lines = [f"Vehicles: {c['vehicle_total']}  ("
                     + " / ".join(f"{l[:1].upper()}B {c[f'vehicle_{l}']}" for l in labels) + ")",
                     f"Pedestrians crossing: {c['ped_total']}",
                     f"Bicycles: {c['bicycle_total']}",
                     f"Over {self.limit:g} mph: {c['over_limit']} of {c['speed_measured']}"
                     + ("" if self.trap_measured else " (est.)")]
            render.draw_panel(frame, lines)


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
