"""Video I/O through the ffmpeg command line.

All timing uses each frame's presentation timestamp (PTS) as reported by
ffmpeg's showinfo filter, never frame_index / fps, because iPhone footage is
variable frame rate.
"""

from __future__ import annotations

import json
import queue
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


class VideoError(RuntimeError):
    pass


def _require(tool: str) -> None:
    if shutil.which(tool) is None:
        raise VideoError(f"{tool} not found on PATH. Install it with: brew install ffmpeg")


def ffprobe_json(path: Path, *args: str) -> dict:
    _require("ffprobe")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-of", "json", *args, str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out)


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float          # nominal rate, used only for the output container
    duration: float
    nb_frames: int | None
    start_pts: float    # stream start_time; subtracted so PTS starts near 0


def probe(path: Path) -> VideoInfo:
    d = ffprobe_json(
        path, "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,start_time:format=duration",
    )
    s = d["streams"][0]
    num, den = (int(x) for x in s["r_frame_rate"].split("/"))
    return VideoInfo(
        width=int(s["width"]),
        height=int(s["height"]),
        fps=num / den,
        duration=float(d["format"]["duration"]),
        nb_frames=int(s["nb_frames"]) if s.get("nb_frames", "N/A").isdigit() else None,
        start_pts=float(s.get("start_time") or 0.0),
    )


def packet_times(path: Path) -> list[float]:
    """Each video packet's timestamp, in file order: the frame order, for a stream without
    B-frames like the camera's."""
    d = ffprobe_json(path, "-select_streams", "v:0", "-show_entries", "packet=pts_time")
    return [float(p["pts_time"]) for p in d.get("packets", []) if p.get("pts_time") not in (None, "N/A")]


def audio_stream_count(path: Path) -> int:
    d = ffprobe_json(path, "-select_streams", "a", "-show_entries", "stream=index")
    return len(d.get("streams", []))


def assert_no_audio(path: Path) -> None:
    n = audio_stream_count(path)
    if n:
        raise VideoError(f"{path} contains {n} audio stream(s); refusing to continue.")


def recording_start(path: Path, tz: str) -> datetime | None:
    """Recording start from container metadata, converted to `tz`.

    Prefers Apple's com.apple.quicktime.creationdate (local time with offset),
    falling back to the generic creation_time (UTC).
    """
    tags = ffprobe_json(path, "-show_entries", "format_tags").get("format", {}).get("tags", {})
    tags = {k.lower(): v for k, v in tags.items()}
    raw = tags.get("com.apple.quicktime.creationdate") or tags.get("creation_time")
    if not raw:
        return None
    raw = raw.replace("Z", "+00:00")
    # "2026-10-15T08:29:01-0700" -> insert colon in the offset for fromisoformat
    raw = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", raw)
    return datetime.fromisoformat(raw).astimezone(ZoneInfo(tz))


def strip_audio(src: Path, dst: Path) -> Path:
    """Copy only the video stream (no audio, data streams or metadata such as GPS).

    This is always the first pipeline step; later steps read only `dst`.
    """
    _require("ffmpeg")
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime and audio_stream_count(dst) == 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".partial" + dst.suffix)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", str(src),
         "-map", "0:v:0", "-c:v", "copy", "-an", "-dn", "-sn",
         "-map_metadata", "-1", "-strict", "unofficial", str(tmp)],
        check=True,
    )
    tmp.replace(dst)
    assert_no_audio(dst)
    return dst


def read_frame_at(path: Path, seconds: float) -> np.ndarray:
    """Decode a single BGR frame at `seconds` (used by calibration)."""
    info = probe(path)
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-ss", f"{seconds:.3f}", "-i", str(path),
         "-frames:v", "1", "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(out, np.uint8).reshape(info.height, info.width, 3).copy()


_PTS_RE = re.compile(r"\bn:\s*\d+\s+pts:\s*-?\d+\s+pts_time:\s*(-?[\d.]+)")


# Decode video stream 0 to raw BGR on stdout, logging each frame's PTS on stderr.
RAW_OUTPUT = ["-map", "0:v:0", "-an", "-sn", "-dn", "-fps_mode", "passthrough", "-vf", "showinfo",
              "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]


class RawFrames:
    """Iterate (pts_seconds, bgr_frame) from an ffmpeg command ending in RAW_OUTPUT."""

    def __init__(self, cmd: list[str], width: int, height: int, **popen_kw):
        _require("ffmpeg")
        self.width, self.height = width, height
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      bufsize=0, **popen_kw)
        self._pts: queue.Queue[float | None] = queue.Queue()
        self._log: list[str] = []
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self._frame_bytes = width * height * 3

    @property
    def log(self) -> list[str]:
        """The last lines ffmpeg printed, other than per-frame info."""
        return list(self._log)

    def _read_stderr(self) -> None:
        for line in iter(self._proc.stderr.readline, b""):
            text = line.decode(errors="replace")
            m = _PTS_RE.search(text)
            if m:
                self._pts.put(float(m.group(1)))
            elif "showinfo" not in text:
                self._log.append(text)
                del self._log[:-50]
        self._pts.put(None)

    def __iter__(self):
        buf = bytearray(self._frame_bytes)
        view = memoryview(buf)
        h, w = self.height, self.width
        while True:
            got = 0
            while got < self._frame_bytes:
                n = self._proc.stdout.readinto(view[got:])
                if not n:
                    break
                got += n
            if got < self._frame_bytes:
                break
            pts = self._pts.get(timeout=30)
            if pts is None:
                raise VideoError("ffmpeg produced a frame without a timestamp")
            frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
            yield pts, frame
        self.close()

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()


class FrameReader(RawFrames):
    """Iterate (pts_seconds, bgr_frame) over a video file.

    `start`/`duration` limit the range (seconds). PTS values are absolute
    within the source (relative to the stream start), even when seeking.
    """

    def __init__(self, path: Path, start: float = 0.0, duration: float | None = None):
        _require("ffmpeg")
        self.info = probe(path)
        cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "info", "-copyts"]
        if start:
            cmd += ["-ss", f"{start:.3f}"]
        if duration:
            cmd += ["-t", f"{duration:.3f}"]
        cmd += ["-i", str(path), *RAW_OUTPUT]
        super().__init__(cmd, self.info.width, self.info.height)

    def __iter__(self):
        for pts, frame in super().__iter__():
            yield pts - self.info.start_pts, frame


class VideoWriter:
    """Encode BGR frames to H.264 MP4 with no audio and no source metadata."""

    def __init__(self, path: Path, width: int, height: int, fps: float,
                 out_height: int | None = None, bitrate: str = "6M"):
        _require("ffmpeg")
        self.path = path
        encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                                  capture_output=True, text=True).stdout
        if "h264_videotoolbox" in encoders:
            codec = ["-c:v", "h264_videotoolbox", "-b:v", bitrate]
        else:
            codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
        vf = ["-vf", f"scale=-2:{out_height}"] if out_height and out_height != height else []
        cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
               "-r", f"{fps:.6f}", "-i", "-", *vf, "-an", *codec,
               "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
               "-color_trc", "bt709", "-map_metadata", "-1", "-movflags", "+faststart",
               str(path)]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self._proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        self._proc.stdin.close()
        if self._proc.wait() != 0:
            raise VideoError(f"ffmpeg encoder failed for {self.path}")
        assert_no_audio(self.path)
