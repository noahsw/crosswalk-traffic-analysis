"""Live video from an RTSP camera such as a TP-Link Tapo C120.

One ffmpeg process per connection reads the stream and does two things with it:

- decodes frames for detection (raw BGR piped to Python), and
- copies the original video, without audio, into short segment files on disk, so a clip of
  any moment from the last few minutes can be cut after the fact.

Both come from the same process, so segment times line up with frame times. Times are
seconds on one session timeline that starts at the first frame. If the stream drops, ffmpeg
is restarted and the new connection continues on the same timeline (by wall clock, so a gap
while the computer slept shows as a gap).
"""

from __future__ import annotations

import csv
import os
import queue
import re
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

from .clips import (clip_name, clip_window, crossing_name, crossing_window, render_clip,
                    render_crossing_clip)
from .video import RAW_OUTPUT, RawFrames, VideoError, _require, ffprobe_json, packet_times
from .tracks import PED_NEAR_PX
from .yields import Crossing, YieldAnalyzer, YieldEvent

USER_ENV, PASSWORD_ENV = "TAPO_USER", "TAPO_PASSWORD"
RTSP_TIMEOUT_US = "10000000"   # give up on a silent stream after 10 s, then reconnect


def camera_url(live: dict, host: str | None = None, stream: str | None = None) -> str:
    """rtsp:// URL for the camera in config.yaml, with the camera account from camera_account()."""
    host = host or live.get("host")
    if not host:
        raise VideoError("No camera address: set live.host in config.yaml or pass --host.")
    user, password = camera_account(live.get("keychain_item") or "tapo-camera")
    return (f"rtsp://{quote(user, safe='')}:{quote(password, safe='')}@{host}:{int(live['port'])}/"
            f"{stream or live['stream']}")


def camera_account(item: str) -> tuple[str, str]:
    """The camera account's (user, password): from TAPO_USER and TAPO_PASSWORD if both are
    set, otherwise from the macOS Keychain item named `item`."""
    user, password = os.environ.get(USER_ENV), os.environ.get(PASSWORD_ENV)
    if user and password:
        return user, password
    found = _keychain_account(item)
    if found:
        return found
    raise VideoError(f"No camera account. Save it in the Keychain with\n"
                     f"  security add-generic-password -U -s {item} -a CAMERA_USER -w\n"
                     f"or set {USER_ENV} and {PASSWORD_ENV} (see the README).")


def _keychain_account(item: str) -> tuple[str, str] | None:
    def security(*args):
        try:
            return subprocess.run(["security", "find-generic-password", "-s", item,
                                   *args], capture_output=True, text=True, timeout=30)
        except (FileNotFoundError, subprocess.TimeoutExpired):   # not a Mac
            return None

    attrs, secret = security(), security("-w")
    if not attrs or attrs.returncode or not secret or secret.returncode:
        return None
    # The account shows as "acct"<blob>="name", or as hex for non-ASCII names.
    m = re.search(r'"acct"<blob>=(?:"(.*)"|0x([0-9A-Fa-f]+))', attrs.stdout)
    if not m:
        return None
    user = m.group(1) if m.group(1) is not None else bytes.fromhex(m.group(2)).decode()
    password = secret.stdout.rstrip("\n")
    return (user, password) if user and password else None


def redact(text: str) -> str:
    """Hide the password in any rtsp:// URL in `text`, for printing."""
    return re.sub(r"(rtsp://[^:/@\s]*):[^@/\s]*@", r"\1:***@", text)


def _input(src: str, realtime: bool = False) -> list[str]:
    if src.startswith("rtsp://"):
        # TCP, because lost UDP packets on Wi-Fi smear the picture.
        return ["-rtsp_transport", "tcp", "-timeout", RTSP_TIMEOUT_US, "-i", src]
    if realtime and "://" not in src:
        return ["-re", "-i", src]   # a recorded file, played back at real speed as if it were live
    return ["-i", src]


def probe_stream(src: str) -> tuple[int, int, float]:
    """(width, height, fps) of a stream's video."""
    opts = ["-rtsp_transport", "tcp", "-timeout", RTSP_TIMEOUT_US] if src.startswith("rtsp://") else []
    try:
        d = ffprobe_json(src, *opts, "-select_streams", "v:0",
                         "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate")
    except subprocess.CalledProcessError as ex:
        raise VideoError(f"Could not open {redact(src)}: {redact((ex.stderr or '').strip())}") from None
    if not d.get("streams"):
        raise VideoError(f"No video in {redact(src)}")
    s = d["streams"][0]
    fps = 15.0
    for key in ("avg_frame_rate", "r_frame_rate"):
        num, _, den = (s.get(key) or "0/0").partition("/")
        if float(den or 0) and 1 <= float(num) / float(den) <= 60:
            fps = float(num) / float(den)
            break
    return int(s["width"]), int(s["height"]), fps


def grab_frame(src: str) -> np.ndarray:
    """One BGR frame from a stream (used by calibration)."""
    _require("ffmpeg")
    try:
        png = subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-nostdin", *_input(src),
                              "-map", "0:v:0", "-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "-"],
                             check=True, capture_output=True, timeout=60).stdout
    except subprocess.CalledProcessError as ex:
        raise VideoError(f"Could not read {redact(src)}: {redact(ex.stderr.decode(errors='replace').strip())}") from None
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)


class SteadyClock:
    """Even frame times from a camera's jittery timestamps.

    The C120 sends a steady 20 frames a second but stamps them unevenly: runs of frames a few
    microseconds apart, then a jump, each stamp off by up to half a second. A car takes about
    half a second to cross the speed trap, so speeds timed from the stamps read anywhere from
    half to several times the real speed. Instead each frame is put one frame period after the
    last, the period measured from the last WINDOW stamps (the stream's stated frame rate can't
    be trusted either), and the clock nudged slightly toward the stamps so it can't drift. On
    the C120 that keeps a half-second interval within about 2.5%.
    """

    GAIN = 0.005
    WINDOW, MIN_FRAMES = 400, 100   # frames: 20 s and 5 s at 20 fps
    MAX_GAP = 1.0                   # seconds between stamps that means frames were lost

    def __init__(self):
        self.stamps: deque[float] = deque(maxlen=self.WINDOW)
        self.t = 0.0

    def __call__(self, pts: float) -> float:
        if self.stamps and abs(pts - self.stamps[-1]) > self.MAX_GAP:
            self.stamps.clear()   # a gap in the stream: start over
        self.stamps.append(pts)
        if len(self.stamps) < self.MIN_FRAMES:
            self.t = pts          # not enough stamps yet to know the frame rate
            return pts
        self.t += (self.stamps[-1] - self.stamps[0]) / (len(self.stamps) - 1)
        self.t += self.GAIN * (pts - self.t)
        return self.t


@dataclass
class Segment:
    path: Path
    start: float   # seconds on the session timeline
    end: float


class LiveSource:
    """Iterate (session_seconds, bgr_frame) from a live stream while recording it in segments.

    If detection falls behind by more than `buffer_seconds`, frames are dropped (counted in
    `dropped`) rather than letting the delay grow. Only the newest `keep_seconds` of recording
    is kept; time when nothing was recorded (asleep, camera offline) doesn't count toward it.
    """

    def __init__(self, src: str, work: Path, width: int, height: int, fps: float, tz,
                 segment_seconds: float = 10.0, keep_seconds: float = 600.0,
                 buffer_seconds: float = 2.0, log=print):
        self.src, self.work = src, work
        self.width, self.height, self.fps, self.tz = width, height, fps, tz
        self.segment_seconds, self.keep_seconds = segment_seconds, keep_seconds
        self.log = log
        self.frames: queue.Queue = queue.Queue(maxsize=max(2, int(round(buffer_seconds * fps))))
        self.start_time: datetime | None = None   # wall clock at session time 0
        self.dropped = 0
        self.reconnects = 0
        self._t0: float | None = None             # time.time() at session time 0
        self._segments: list[Segment] = []        # from finished connections
        # (camera timestamp, session time) of each recent frame, both on the session timeline,
        # to give clip frames the times their tracks have (see frame_times).
        self._times: deque[tuple[float, float]] = deque(maxlen=int(fps * (keep_seconds + 60)))
        self._open: tuple[Path, float] | None = None   # (folder, time offset) of the current connection
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        work.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def lag_seconds(self) -> float:
        return self.frames.qsize() / self.fps

    def _command(self, folder: Path) -> list[str]:
        return ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "info", *_input(self.src, realtime=True),
                "-map", "0:v:0", "-an", "-sn", "-dn", "-c:v", "copy",
                "-f", "segment", "-segment_time", f"{self.segment_seconds:g}",
                "-segment_format", "mpegts", "-reset_timestamps", "1",
                "-segment_list", str(folder / "segments.csv"), "-segment_list_type", "csv",
                "-segment_list_flags", "+live", str(folder / "%06d.ts"),
                *RAW_OUTPUT]

    def _run(self) -> None:
        conn = 0
        while not self._stop.is_set():
            folder = self.work / f"connection_{conn:03d}"
            folder.mkdir(parents=True, exist_ok=True)
            # Its own session, so Ctrl+C reaches only Python, which then stops ffmpeg cleanly.
            reader = RawFrames(self._command(folder), self.width, self.height, start_new_session=True)
            self._proc = reader._proc
            if self._stop.is_set():
                self._proc.send_signal(signal.SIGINT)
            offset = None
            next_prune = 0.0
            error = ""
            clock = SteadyClock()
            try:
                for stamp, frame in reader:
                    pts = clock(stamp)
                    if offset is None:
                        # Wall clock, not time.monotonic(): on a Mac that stops while asleep.
                        now = time.time()
                        if self._t0 is None:
                            self._t0 = now - pts
                            self.start_time = datetime.now(self.tz) - timedelta(seconds=pts)
                        offset = now - self._t0 - pts
                        with self._lock:
                            self._open = (folder, offset)
                    t = pts + offset
                    with self._lock:
                        self._times.append((stamp + offset, t))
                    try:
                        self.frames.put_nowait((t, frame))
                    except queue.Full:
                        self.dropped += 1
                    if t >= next_prune:
                        self._prune()
                        next_prune = t + self.segment_seconds
            except VideoError as ex:
                error = str(ex)
            reader.close()
            with self._lock:
                if offset is not None:
                    self._segments += self._read_list(folder, offset)
                self._open = None
            if self._stop.is_set() or "://" not in self.src:
                break   # stopped, or a recorded file reached its end
            self.reconnects += 1
            lines = [line.strip() for line in reader.log if "Opening '" not in line][-2:]
            why = " | ".join(lines + ([error] if error else []))
            why = redact(why) or "no error message"
            self.log(f"Camera stream ended ({why}). Reconnecting in 5 s...")
            self._stop.wait(5)
            conn += 1
        while not self._stop.is_set():
            try:
                self.frames.put(None, timeout=1)
                break
            except queue.Full:
                pass

    def __iter__(self):
        while True:
            try:
                item = self.frames.get(timeout=1)
            except queue.Empty:
                if not self._thread.is_alive():
                    return
                continue
            if item is None:
                return
            yield item

    def stop(self) -> None:
        """Stop reading. ffmpeg finishes the segment it is writing, so it can still be cut."""
        self._stop.set()
        proc = self._proc
        if proc and proc.poll() is None:
            proc.send_signal(signal.SIGINT)
        self._thread.join(timeout=15)
        if proc and proc.poll() is None:
            proc.kill()
            self._thread.join(timeout=5)

    # ------------------------------------------------------------------ recording
    @staticmethod
    def _read_list(folder: Path, offset: float) -> list[Segment]:
        path = folder / "segments.csv"
        if not path.exists():
            return []
        lines = path.read_text().split("\n")[:-1]   # the last line may still be being written
        return [Segment(folder / row[0], float(row[1]) + offset, float(row[2]) + offset)
                for row in csv.reader(lines) if len(row) == 3]

    def segments(self) -> list[Segment]:
        """Finished segments still on disk, oldest first."""
        with self._lock:
            segs = list(self._segments)
            if self._open:
                segs += self._read_list(*self._open)
        return [s for s in segs if s.path.exists()]

    def _prune(self) -> None:
        kept = 0.0
        for s in reversed(self.segments()):
            if kept > self.keep_seconds:
                s.path.unlink(missing_ok=True)
            kept += s.end - s.start
        with self._lock:
            self._segments = [s for s in self._segments if s.path.exists()]

    def frame_times(self, piece: Path, start: float) -> list[float] | None:
        """The session time of each frame in `piece` (cut to start at session time `start`), as
        the frames were timed for tracking, or None if they can't be matched up.

        The recording keeps the camera's own timestamps, up to half a second off the times the
        tracks use, so a clip timed by them draws boxes ahead of or behind the vehicles. The
        frames are found among the recently seen ones by those same timestamps: their uneven
        pattern only lines up one way.
        """
        stamps = np.asarray(packet_times(piece))
        with self._lock:
            seen = np.asarray(self._times)
        n = min(len(stamps), 100)
        if n < 20 or len(seen) < n:
            return None
        piece_stamps = start + stamps - stamps[0]
        k0 = int(np.argmin(np.abs(seen[:, 0] - piece_stamps[0])))
        shifts = range(max(0, k0 - 100), min(len(seen) - n, k0 + 100) + 1)
        if not shifts:
            return None
        err, k = min((float(np.median(np.abs(seen[s:s + n, 0] - piece_stamps[:n]))), s) for s in shifts)
        if err > 0.01:
            return None
        times = list(seen[k:k + len(stamps), 1])
        period = (seen[-1, 1] - seen[-n, 1]) / (n - 1)
        while len(times) < len(stamps):   # frames not seen yet: one frame period apart
            times.append(times[-1] + period)
        return times

    def cut(self, t0: float, t1: float, dst: Path) -> float | None:
        """Copy the recording from session time t0 to t1 into `dst`.

        Returns the session time `dst` starts at, or None if the recording hasn't reached t1 yet.
        """
        segs = self.segments()
        if not self._stop.is_set() and not any(s.end >= t1 for s in segs):
            return None
        use = [s for s in segs if s.end >= t0 and s.start <= t1]
        if not use:
            raise VideoError(f"no recording of {t0:.0f}-{t1:.0f} s (already deleted, or the stream was down)")
        # Pieces from different connections have a gap between them, so take only the last one's.
        use = [s for s in use if s.path.parent == use[-1].path.parent]
        listing = dst.with_suffix(".txt")
        listing.write_text("".join(f"file '{s.path.resolve()}'\n" for s in use))
        try:
            subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "concat", "-safe", "0",
                            "-i", str(listing), "-map", "0:v:0", "-c", "copy", "-an", str(dst)], check=True)
        finally:
            listing.unlink()
        return use[0].start


@dataclass
class JudgedCrossing:
    crossing: Crossing
    events: list[YieldEvent]   # drivers judged for the first time on this crossing
    vehicles: list             # every vehicle that could have been near it, for its clip


class LiveYieldWatcher:
    """Judges each pedestrian crossing shortly after it ends, with the same rules as yielding.py.

    A crossing is judged SETTLE_SECONDS after the pedestrian was last seen. By then every
    vehicle that can count against it (reaching the crosswalk up to 10 s after they finished)
    has passed, and has been tracked a little further to settle its direction.

    A driver is judged against the first of the pedestrians it passed (as in yielding.py), so a
    crossing also waits while anyone who appeared before them near the crosswalk is still being
    tracked or waiting to be judged, then they're judged together. Otherwise someone who stepped
    up just as a car arrived, and left first, would have it "too close to stop", though the
    person who'd been waiting beside them for several seconds was not yielded to.
    """

    SETTLE_SECONDS = 15.0
    MAX_HOLD_SECONDS = 120.0   # ...but never longer than this, for someone loitering nearby

    def __init__(self, cfg: dict, cal: dict, analyzer):
        self.analyzer = analyzer
        self.ya = YieldAnalyzer(cfg, cal, analyzer.direction_of)
        self._seen = 0                          # how many of analyzer.ped_records are in _waiting
        self._waiting: list = []                # pedestrians not judged yet
        self._judged: set[int] = set()   # vehicle track ids, so each driver is judged once

    def poll(self, now: float, force: bool = False) -> list[JudgedCrossing]:
        a = self.analyzer
        self._waiting += a.ped_records[self._seen:]
        self._seen = len(a.ped_records)
        settled = {id(p) for p in self._waiting if force or p.last_pts + self.SETTLE_SECONDS <= now}
        earlier = min([p.first_pts for p in self._waiting if id(p) not in settled]
                      + [t.first_pts for t in a.people.values() if t.closest_px >= -PED_NEAR_PX],
                      default=float("inf"))
        ready = [p for p in self._waiting if id(p) in settled and (
                 force or p.first_pts < earlier
                 or p.last_pts + self.SETTLE_SECONDS + self.MAX_HOLD_SECONDS <= now)]
        if not ready:
            return []
        self._waiting = [p for p in self._waiting if not any(p is r for r in ready)]
        since = min(p.first_pts for p in ready)   # vehicles must reach the crosswalk after this
        vehicles = [v for v in a.vehicles.values() if v.relevant]
        for v in reversed(a.vehicle_records):     # roughly in order of when they were last seen
            if v.last_pts < since - 60:
                break
            if v.last_pts >= since:
                vehicles.append(v)
        judged = []
        for c in self.ya.analyze(ready, vehicles):
            new = [e for e in c.events if e.crossing is c and e.vehicle.tid not in self._judged]
            self._judged.update(e.vehicle.tid for e in new)
            judged.append(JudgedCrossing(c, new, vehicles))
        return judged


@dataclass
class _Clip:
    clip_dir: Path
    events: list[YieldEvent] = field(default_factory=list)   # drivers passing one pedestrian, or
    group: list[JudgedCrossing] = field(default_factory=list)   # pedestrians crossing together
    vehicles: list = field(default_factory=list)       # for `events`: every vehicle near them
    not_before: float = 0.0                            # time.time() to wait for, so others can join
    started: bool = False                              # being cut, so nothing more joins it


class ClipSaver:
    """Cuts and renders clips in the background, once the recording covers them."""

    CROSSING_SECONDS_AFTER = 3.0
    TOGETHER_SECONDS = 2.0   # crossings this close in time share one clip...
    HOLD_SECONDS = 10.0      # ...if judged within this long of each other

    def __init__(self, source: LiveSource, cal: dict, start_time, seconds_before: float,
                 seconds_after: float, crossing_seconds_before: float = 10.0, log=print):
        self.source, self.cal, self.start_time = source, cal, start_time
        self.before, self.after = seconds_before, seconds_after
        self.crossing_before = crossing_seconds_before
        self.log = log
        self.saved = 0
        self._pending: list[_Clip] = []
        self._closing = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def add(self, events: list[YieldEvent], vehicles: list, clip_dir: Path) -> Path:
        """Queue one clip of the drivers in `events` passing one pedestrian (see by_pedestrian).
        Returns where it will be saved."""
        clip_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._pending.append(_Clip(clip_dir, events=events, vehicles=vehicles))
        return clip_dir / clip_name(self.start_time, events)

    def add_crossing(self, j: JudgedCrossing, clip_dir: Path) -> None:
        """Queue the clip of a whole pedestrian crossing. Pedestrians who cross together go in
        one clip, named when it is saved."""
        clip_dir.mkdir(parents=True, exist_ok=True)
        c, gap = j.crossing, self.TOGETHER_SECONDS
        with self._lock:
            for clip in self._pending:
                if (clip.group and not clip.started and clip.clip_dir == clip_dir
                        and any(c.wait_start <= o.crossing.exit + gap
                                and o.crossing.wait_start <= c.exit + gap for o in clip.group)):
                    clip.group.append(j)
                    return
            self._pending.append(_Clip(clip_dir, group=[j], not_before=time.time() + self.HOLD_SECONDS))

    def close(self) -> None:
        """Save everything still pending (call after source.stop()), then finish."""
        with self._lock:
            self._closing = True
        self._thread.join()

    def _run(self) -> None:
        while True:
            with self._lock:
                todo, closing = list(self._pending), self._closing
            if not todo and closing:
                return
            for clip in todo:
                if clip.not_before > time.time() and not closing:
                    continue
                with self._lock:
                    clip.started = True
                if self._save(clip):
                    with self._lock:
                        self._pending.remove(clip)
                else:
                    with self._lock:
                        clip.started = False
            time.sleep(1)

    def _save(self, clip: _Clip) -> bool:
        """Try to save one clip. False if the recording doesn't cover it yet."""
        if clip.events:
            events = clip.events
            path = clip.clip_dir / clip_name(self.start_time, events)
            window = clip_window(events, self.before, self.after)

            def render(piece, offset, times):
                render_clip(piece, path, events, clip.vehicles, self.cal["crosswalk"],
                            self.start_time, self.before, self.after, offset=offset, times=times)
        else:
            crossings = [j.crossing for j in clip.group]
            vehicles = list({id(v): v for j in clip.group for v in j.vehicles}.values())
            before, after = self.crossing_before, self.CROSSING_SECONDS_AFTER
            path = clip.clip_dir / crossing_name(self.start_time, crossings)
            window = crossing_window(crossings, before, after)

            def render(piece, offset, times):
                render_crossing_clip(piece, path, crossings, vehicles, self.cal["crosswalk"],
                                     self.start_time, before, after, offset=offset, times=times)
        piece = self.source.work / f"{path.stem}.mkv"
        try:
            offset = self.source.cut(*window, piece)
            if offset is None:
                return False
            times = self.source.frame_times(piece, offset)
            if times is None:
                self.log(f"{path.name}: couldn't match its frames to the tracks' times, so its "
                         "boxes may be up to half a second off")
            render(piece, offset, times)
            self.saved += 1
            self.log(f"Saved {path}")
        except (VideoError, subprocess.CalledProcessError) as ex:
            self.log(f"Could not save {path.name}: {ex}")
        finally:
            piece.unlink(missing_ok=True)
        return True
