import subprocess

import numpy as np

from traffic.clips import copy_window
from traffic.live import SteadyClock
from traffic.video import FrameReader, audio_stream_count


def _stamps(fps, seconds, seed=0):
    """A steady frame rate, stamped like the C120 does: in clumps, each off by up to ~0.4 s."""
    rng = np.random.default_rng(seed)
    true = np.arange(int(fps * seconds)) / fps
    stamps = true + rng.uniform(-0.2, 0.2, len(true))
    clumps = np.maximum.accumulate(np.where(rng.random(len(true)) < 0.4, 0, stamps))
    return true, np.maximum(stamps, clumps)


def _timed(true, stamps):
    clock = SteadyClock()
    return np.array([clock(s) for s in stamps])


def test_half_second_intervals_are_even():
    for fps in (20, 15):
        true, stamps = _stamps(fps, 300)
        t = _timed(true, stamps)[400:]
        n = fps // 2
        ratio = (t[n:] - t[:-n]) / (true[400:][n:] - true[400:][:-n])
        assert np.all(np.abs(ratio - 1) < 0.05)


def test_starts_over_after_a_gap():
    clock = SteadyClock()
    for i in range(200):
        clock(i / 20)
    assert clock(60.0) == 60.0


def test_raw_copy_is_the_window_as_recorded(tmp_path):
    """A crossing's raw copy (see ClipSaver.add_crossing) covers its window, with no audio."""
    piece = tmp_path / "piece.mkv"
    subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=20",
                    "-f", "lavfi", "-i", "sine", "-t", "10", "-c:v", "mpeg4", "-g", "10", str(piece)], check=True)
    raw = tmp_path / "raw_crossings" / "crossing.mp4"
    copy_window(piece, raw, (104.0, 107.0), offset=100.0)   # the piece starts 100 s into the session
    times = [pts for pts, _ in FrameReader(raw)]
    assert len(times) == 60 and times[0] == 0
    assert audio_stream_count(raw) == 0
