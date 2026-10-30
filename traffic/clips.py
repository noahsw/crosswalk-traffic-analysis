"""Short captioned video clips of yielding events (and, live, of every pedestrian crossing)."""

from __future__ import annotations

import subprocess
from collections import Counter
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


def clip_name(start_time, e: YieldEvent) -> str:
    ts = start_time + timedelta(seconds=e.passes)
    return f"{ts:%Y%m%d_%H%M%S}_{e.outcome}_{e.direction or 'unknown'}_V{e.vehicle.tid}.mp4"


def clip_window(e: YieldEvent, seconds_before: float, seconds_after: float) -> tuple[float, float]:
    """(start, end) of an event's clip, in seconds on the tracks' timeline."""
    c = e.crossing
    t0 = max(0.0, min(e.passes - 4.0, max(c.wait_start - 1.0, e.passes - seconds_before)))
    return t0, e.passes + seconds_after


def render_clip(video: Path, out_path: Path, e: YieldEvent, crosswalk, start_time,
                seconds_before: float, seconds_after: float, offset: float = 0.0) -> None:
    """`offset` is where `video` starts on the tracks' timeline (for a piece of a live stream)."""
    t0, t1 = clip_window(e, seconds_before, seconds_after)
    c = e.crossing
    reader = FrameReader(video, start=max(0.0, t0 - offset), duration=t1 - t0)
    info = reader.info
    writer = VideoWriter(out_path, info.width, info.height, info.fps, bitrate="5M")
    cw = np.asarray(crosswalk, np.int32)
    caption = caption_for(e)
    color = render.RED if e.outcome == "did_not_yield" else render.GREEN
    for pts, frame in reader:
        pts += offset
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


# Crossing clips cover a group of pedestrians who crossed together (often just one).

def _group_outcomes(crossings) -> Counter:
    judged = {e.vehicle.tid: e.outcome for c in crossings for e in c.events}
    return Counter(judged.values())


def crossing_caption(crossings) -> str:
    """e.g. "Pedestrian crossed northbound after waiting 6 s: 2 drivers yielded"."""
    waits = max(c.wait_s for c in crossings)
    who = (f"{len(crossings)} pedestrians" if len(crossings) > 1 else "Pedestrian")
    directions = " and ".join(sorted({c.direction for c in crossings}))
    waited = (f" after waiting {'up to ' if len(crossings) > 1 else ''}{waits:.0f} s"
              if waits >= 1 else "")
    n = _group_outcomes(crossings)
    drivers = ", ".join(f"{n[o]} {'driver' if n[o] == 1 else 'drivers'} {OUTCOME_TEXT[o]}"
                        for o in OUTCOME_TEXT if n[o]) or "no drivers needed to yield"
    return f"{who} crossed {directions}{waited}: {drivers}"


def crossing_name(start_time, crossings) -> str:
    first = min(crossings, key=lambda c: c.wait_start)
    ts = start_time + timedelta(seconds=first.wait_start)
    peds = "+".join(f"P{c.ped.tid}" for c in sorted(crossings, key=lambda c: c.ped.tid))
    n = _group_outcomes(crossings)
    tag = "".join(f"_{n[o]}{o}" for o in OUTCOME_TEXT if n[o])
    return f"{ts:%Y%m%d_%H%M%S}_{first.direction}_{peds}{tag}.mp4"


def crossing_window(crossings, seconds_before: float, seconds_after: float) -> tuple[float, float]:
    """(start, end) of a crossing clip: from before the pedestrians started trying to cross until
    they're across and every driver judged against them has reached the crosswalk."""
    start = min(c.wait_start for c in crossings)
    end = max([c.exit for c in crossings] + [e.passes for c in crossings for e in c.events])
    return max(0.0, start - seconds_before), end + seconds_after


def render_crossing_clip(video: Path, out_path: Path, crossings, vehicles, crosswalk, start_time,
                         seconds_before: float, seconds_after: float, offset: float = 0.0) -> None:
    """Every vehicle in view is boxed: judged drivers by outcome (red = did not yield), others gray."""
    t0, t1 = crossing_window(crossings, seconds_before, seconds_after)
    judged = {e.vehicle.tid: e.outcome for c in crossings for e in c.events}
    vehicles = [v for v in vehicles if v.kind == "vehicle" and v.boxes
                and v.boxes[0][0] <= t1 and v.boxes[-1][0] >= t0]
    reader = FrameReader(video, start=max(0.0, t0 - offset), duration=t1 - t0)
    info = reader.info
    writer = VideoWriter(out_path, info.width, info.height, info.fps, bitrate="5M")
    cw = np.asarray(crosswalk, np.int32)
    caption = crossing_caption(crossings)
    for pts, frame in reader:
        pts += offset
        layer = frame.copy()
        cv2.fillPoly(layer, [cw], render.CROSSWALK_COLOR)
        cv2.addWeighted(layer, 0.3, frame, 0.7, 0, dst=frame)
        for v in vehicles:
            vb = box_at(v.boxes, pts)
            if vb is None:
                continue
            outcome = judged.get(v.tid)
            if outcome:
                color = render.RED if outcome == "did_not_yield" else render.GREEN
                render.draw_box(frame, vb, color, f"#{v.tid} {OUTCOME_TEXT[outcome]}", thickness=4)
            else:
                render.draw_box(frame, vb, render.GRAY, f"#{v.tid}")
        for c in crossings:
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
