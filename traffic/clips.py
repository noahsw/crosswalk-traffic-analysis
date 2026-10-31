"""Short captioned video clips of yielding events (and, live, of every pedestrian crossing)."""

from __future__ import annotations

import subprocess
from collections import Counter
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np

from . import render
from .video import FrameReader, VideoWriter, assert_no_audio, probe
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


def live_clip_dir(session_dir: Path, outcome: str) -> Path:
    """Where live sessions save their driver clips of one outcome: a single folder beside the
    session folders (out/did_not_yield/), so the clips from every session are in one place."""
    return session_dir.parent / outcome


def by_pedestrian(events) -> list[list[YieldEvent]]:
    """Drivers' events grouped by the pedestrian they were judged against, one group per clip.
    Pedestrians crossing together share their drivers, so they share a group. Groups and the
    events in them are in the order the drivers reached the crosswalk."""
    groups: list[list[YieldEvent]] = []
    for e in sorted(events, key=lambda e: e.passes):
        peds = {id(c) for c in e.crossings}
        joins = [g for g in groups if any(id(c) in peds for x in g for c in x.crossings)]
        groups = [g for g in groups if not any(g is j for j in joins)]
        groups.append(sorted([x for g in joins for x in g] + [e], key=lambda x: x.passes))
    return sorted(groups, key=lambda g: g[0].passes)


def clip_name(start_time, events: list[YieldEvent]) -> str:
    """e.g. 20261030_143933_did_not_yield_westbound+eastbound_V463+V464.mp4, for the events of
    one outcome against one pedestrian (see by_pedestrian)."""
    events = sorted(events, key=lambda e: e.passes)
    ts = start_time + timedelta(seconds=events[0].passes)
    directions = "+".join(dict.fromkeys(e.direction or "unknown" for e in events))
    vehicles = "+".join(f"V{e.vehicle.tid}" for e in events)
    return f"{ts:%Y%m%d_%H%M%S}_{events[0].outcome}_{directions}_{vehicles}.mp4"


def clip_window(events: list[YieldEvent], seconds_before: float,
                seconds_after: float) -> tuple[float, float]:
    """(start, end) of a clip of drivers passing one pedestrian, in seconds on the tracks'
    timeline: from a little before the first reaches the crosswalk until just after the last has
    and the pedestrians have made it across."""
    first = min(events, key=lambda e: e.passes)
    t0 = max(0.0, min(first.passes - 4.0, max(first.crossing.wait_start - 1.0,
                                              first.passes - seconds_before)))
    end = max([e.passes for e in events] + [c.exit for e in events for c in e.crossings])
    return t0, end + seconds_after


def _frames(video: Path, t0: float, t1: float, offset: float, times: list[float] | None):
    """(time on the tracks' timeline, frame) from t0 to t1.

    `offset` is where `video` starts on the tracks' timeline (for a piece of a live stream).
    `times`, if given, is the time of each of its frames as the tracks timed them, for video
    whose own timestamps were replaced (see live.SteadyClock).
    """
    if times is None:
        for pts, frame in FrameReader(video, start=max(0.0, t0 - offset), duration=t1 - t0):
            yield pts + offset, frame
        return
    for t, (_, frame) in zip(times, FrameReader(video)):
        if t0 <= t <= t1:
            yield t, frame


def render_clip(video: Path, out_path: Path, events: list[YieldEvent], vehicles, crosswalk,
                start_time, seconds_before: float, seconds_after: float, offset: float = 0.0,
                times: list[float] | None = None) -> None:
    """A clip of the drivers in `events` passing one pedestrian (see by_pedestrian).
    `offset` and `times`: see _frames."""
    crossings = list({id(c): c for e in events for c in e.crossings}.values())
    _render(video, out_path, crossings, vehicles, crosswalk, start_time,
            clip_window(events, seconds_before, seconds_after), offset, times)


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
                         seconds_before: float, seconds_after: float, offset: float = 0.0,
                         times: list[float] | None = None) -> None:
    """`offset` and `times`: see _frames."""
    _render(video, out_path, crossings, vehicles, crosswalk, start_time,
            crossing_window(crossings, seconds_before, seconds_after), offset, times)


def _render(video: Path, out_path: Path, crossings, vehicles, crosswalk, start_time,
            window: tuple[float, float], offset: float, times: list[float] | None) -> None:
    """Every vehicle in view is boxed: judged drivers by outcome (red = did not yield), others gray."""
    t0, t1 = window
    judged = {e.vehicle.tid: e for c in crossings for e in c.events}
    vehicles = [v for v in vehicles if v.kind == "vehicle" and v.boxes
                and v.boxes[0][0] <= t1 and v.boxes[-1][0] >= t0]
    info = probe(video)
    writer = VideoWriter(out_path, info.width, info.height, info.fps, bitrate="5M")
    cw = np.asarray(crosswalk, np.int32)
    caption = crossing_caption(crossings)
    for pts, frame in _frames(video, t0, t1, offset, times):
        layer = frame.copy()
        cv2.fillPoly(layer, [cw], render.CROSSWALK_COLOR)
        cv2.addWeighted(layer, 0.3, frame, 0.7, 0, dst=frame)
        for v in vehicles:
            vb = box_at(v.boxes, pts)
            if vb is None:
                continue
            e = judged.get(v.tid)
            if e:
                color = render.RED if e.outcome == "did_not_yield" else render.GREEN
                speed = f" {e.speed_mph:.0f} mph" if e.outcome == "did_not_yield" and e.speed_mph else ""
                render.draw_box(frame, vb, color, f"#{v.tid} {OUTCOME_TEXT[e.outcome]}{speed}",
                                thickness=4)
            else:
                render.draw_box(frame, vb, render.GRAY, f"#{v.tid}")
        for c in crossings:
            pb = box_at(c.ped.boxes, pts)
            if pb is not None:
                render.draw_box(frame, pb, render.MAGENTA, f"P{c.ped.tid}", thickness=3)
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
