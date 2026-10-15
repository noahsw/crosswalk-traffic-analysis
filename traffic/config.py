"""Load config.yaml (settings) and calibration.yaml (scene geometry)."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "speed_limit_mph": 25,
    "timezone": "America/Los_Angeles",
    "direction_labels": {"left_to_right": "left_to_right", "right_to_left": "right_to_left"},
    "crosswalk_direction_labels": {"near_to_far": "near_to_far", "far_to_near": "far_to_near"},
    "road": {
        "lanes": 2,
        "crosswalk_length_ft": 36,
        "crosswalk_length_measured": False,
        "walking_speed_fps": 3.5,
        "startup_seconds": 3.0,
        "existing_treatments": [],
    },
    "detection": {
        "model": "models/yolo11m.pt",
        "imgsz": 960,
        "confidence": 0.25,
        "device": "auto",
        "stride": 2,
        "vehicle_classes": ["car", "motorcycle", "bus", "truck"],
        "bicycle_classes": ["bicycle"],
        "person_classes": ["person"],
    },
    "tracking": {
        "lost_track_seconds": 1.5,
        "finalize_after_seconds": 3.0,
        "smoothing_samples": 5,
    },
    "speed": {"min_points": 4, "min_mph": 3, "max_mph": 80},
    "pedestrians": {
        "min_seconds_in_crosswalk": 0.5,
        "min_crossing_fraction": 0.3,
        "max_containment_in_vehicle": 0.9,
    },
    "yielding": {
        "enabled": True,
        "lanes": {},
        "min_seconds_to_stop": 3.0,
        "stopped_mph": 5.0,
        "waiting_area_px": 90,
        "clip_seconds_before": 6.0,
        "clip_seconds_after": 3.0,
        "max_clips": 60,
    },
    "output": {
        "video": True,
        "height": None,
        "bitrate": "6M",
        "overlays": {"zones": True, "gates": True, "timestamp": True, "counters": True,
                     "trails": False},
        "movement_check_samples": 3,
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path | None) -> dict:
    user = yaml.safe_load(Path(path).read_text()) if path and Path(path).exists() else {}
    return _merge(DEFAULTS, user or {})


REQUIRED_CALIBRATION = ["count_line", "gates", "crosswalk", "speed_trap"]


def load_calibration(path: Path) -> dict:
    cal = yaml.safe_load(Path(path).read_text())
    missing = [k for k in REQUIRED_CALIBRATION if k not in cal]
    if missing:
        raise ValueError(f"{path} is missing: {', '.join(missing)}. Run calibrate.py.")
    if len(cal["crosswalk"]) != 4:
        raise ValueError("crosswalk must have 4 points: near-side 2 then far-side 2")
    return cal


def save_calibration(cal: dict, path: Path) -> None:
    def flow_points(obj):
        # Keep [x, y] pairs on one line for readability.
        if isinstance(obj, dict):
            return {k: flow_points(v) for k, v in obj.items()}
        if isinstance(obj, list) and obj and all(isinstance(p, list) and len(p) == 2 for p in obj):
            return [FlowList(p) for p in obj]
        return obj

    class FlowList(list):
        pass

    class Dumper(yaml.SafeDumper):
        pass

    Dumper.add_representer(
        FlowList, lambda d, v: d.represent_sequence("tag:yaml.org,2002:seq", v, flow_style=True))
    Path(path).write_text(yaml.dump(flow_points(cal), Dumper=Dumper, sort_keys=False))
