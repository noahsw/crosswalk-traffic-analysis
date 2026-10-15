"""Short captioned video clips of yielding events, cut from the audio-free working copy."""

from __future__ import annotations

import subprocess
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np

from . import render
from .video import FrameReader, VideoWriter, assert_no_audio
from .yields import YieldEvent, box_at

OUTCOME_TEXT = {
    "did_not_yield": "did not yield",
    "yielded": "yielded",
    "too_close_to_stop": "too close to stop",
}


def caption_for(e: YieldEvent) -> str:
    who = {"waiting_at_curb": "a pedestrian waiting to cross",
           "in_crosswalk": "a pedestrian in the crosswalk"}[e.phase]
    speed = f" at about {e.speed_mph:.0f} mph" if e.speed_mph is not None else ""
    if e.outcome == "did_not_yield":
        return f"{e.direction.capitalize()} vehicle did not yield to {who}{speed}"
    if e.outcome == "yielded":
        return f"{e.direction.capitalize()} vehicle stopped for {who}"
    return f"{e.direction.capitalize()} vehicle arrived too soon to stop for {who}"


def render_clip(video: Path, out_path: Path, e: YieldEvent, crosswalk, start_time,
                seconds_before: float, seconds_after: float) -> None:
    c = e.crossing
    t0 = max(0.0, min(e.passes - 4.0, max(c.wait_start - 1.0, e.passes - seconds_before)))
    t1 = e.passes + seconds_after
    reader = FrameReader(video, start=t0, duration=t1 - t0)
    info = reader.info
    writer = VideoWriter(out_path, info.width, info.height, info.fps, bitrate="5M")
    cw = np.asarray(crosswalk, np.int32)
    caption = caption_for(e)
    color = render.RED if e.outcome == "did_not_yield" else render.GREEN
    for pts, frame in reader:
        layer = frame.copy()
        cv2.fillPoly(layer, [cw], render.CROSSWALK_COLOR)
        cv2.addWeighted(layer, 0.3, frame, 0.7, 0, dst=frame)
        vb = box_at(e.vehicle.boxes, pts)
        if vb is not None:
            render.draw_box(frame, vb, color, f"#{e.vehicle.tid} {OUTCOME_TEXT[e.outcome]}", thickness=4)
        pb = box_at(c.ped.boxes, pts)
        if pb is not None:
            render.draw_box(frame, pb, render.MAGENTA, "pedestrian", thickness=3)
        ts = start_time + timedelta(seconds=pts)
        render.draw_timestamp(frame, ts.strftime("%a %b %d %Y %H:%M:%S PT"))
        render.put_label(frame, caption, (12, frame.shape[0] - 14), render.WHITE, scale=1.0,
                         thickness=2, pad=10)
        writer.write(frame)
    writer.close()


def concat(clips: list[Path], out_path: Path) -> None:
    """Join clips (same encoding settings) into one video without re-encoding."""
    if not clips:
        return
    listing = out_path.with_suffix(".txt")
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in clips))
    subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(listing), "-c", "copy", "-an", "-movflags", "+faststart", str(out_path)],
                   check=True)
    listing.unlink()
    assert_no_audio(out_path)
