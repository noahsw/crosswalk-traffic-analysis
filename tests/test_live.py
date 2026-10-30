import numpy as np

from traffic.live import SteadyClock


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
