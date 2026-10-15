"""Published reference points for putting the counts in context, plus crossing-gap analysis.

Sources (cited in summary.md):
- FHWA (2018). Guide for Improving Pedestrian Safety at Uncontrolled Crossing Locations
  (STEP guide), Table 1.
- FHWA (2023). Highway Functional Classification Concepts, Criteria and Procedures:
  typical urban AADT ranges.
- Tefft, B.C. (2011). Impact Speed and a Pedestrian's Risk of Severe Injury or Death.
  AAA Foundation for Traffic Safety.
- Highway Capacity Manual: pedestrian delay level of service at unsignalized crossings.
"""

from __future__ import annotations

import math

import numpy as np

# Share of daily traffic in the busiest hour (rule of thumb used to rough out daily volume).
PEAK_HOUR_SHARE = (0.08, 0.12)

# FHWA typical urban AADT ranges by functional class.
FUNCTIONAL_CLASSES = [
    ("Local street", 80, 700),
    ("Collector", 1_100, 6_300),
    ("Minor arterial", 3_000, 14_000),
    ("Principal arterial", 7_000, 27_000),
]

STEP_COUNTERMEASURES = {
    1: "High-visibility crosswalk, parking restrictions, lighting and warning signs",
    2: "Raised crosswalk",
    3: "Advance yield/stop sign and line",
    4: "In-street pedestrian crossing sign",
    5: "Curb extension",
    6: "Pedestrian refuge island",
    7: "Rectangular Rapid-Flashing Beacon (RRFB)",
    8: "Road diet",
    9: "Pedestrian Hybrid Beacon (PHB)",
}

# STEP guide Table 1, row "2 lanes (1 lane in each direction)".
# Keyed by AADT band, then posted-speed band.
STEP_TABLE_2_LANES = {
    "<9,000": {"<=30": [1, 2, 4, 5, 6], "35": [1, 5, 6, 7, 9], ">=40": [1, 5, 6, 7, 9]},
    "9,000-15,000": {"<=30": [1, 4, 5, 6], "35": [1, 5, 6, 7, 9], ">=40": [1, 5, 6, 7, 9]},
    ">15,000": {"<=30": [1, 4, 5, 6, 7, 9], "35": [1, 5, 6, 7, 9], ">=40": [1, 5, 6, 9]},
}

# Tefft (2011): impact speed (mph) at which the average pedestrian's risk reaches each level.
DEATH_RISK_MPH = [(10, 23), (25, 32), (50, 42), (75, 50), (90, 58)]
SEVERE_INJURY_RISK_MPH = [(10, 16), (25, 23), (50, 31), (75, 39), (90, 46)]

# HCM pedestrian LOS at unsignalized crossings by average delay (seconds).
PED_LOS = [("A", 5), ("B", 10), ("C", 20), ("D", 30), ("E", 45)]


def aadt_band(aadt: float) -> str:
    if aadt < 9_000:
        return "<9,000"
    return "9,000-15,000" if aadt <= 15_000 else ">15,000"


def speed_band(mph: float) -> str:
    # Table columns are posted limits in 5 mph steps; round an operating speed to the nearest 5.
    mph = 5 * round(mph / 5)
    if mph <= 30:
        return "<=30"
    return "35" if mph < 40 else ">=40"


def functional_class(aadt: float) -> list[str]:
    return [name for name, lo, hi in FUNCTIONAL_CLASSES if lo <= aadt <= hi]


def ped_los(delay_s: float) -> str:
    for grade, upper in PED_LOS:
        if delay_s <= upper:
            return grade
    return "F"


def risk_at(mph: float, table) -> float:
    """Linear interpolation of the Tefft risk curve (percent) at `mph`."""
    pts = [(0, 0)] + [(s, r) for r, s in table] + [(80, 100)]
    for (s0, r0), (s1, r1) in zip(pts, pts[1:]):
        if s0 <= mph <= s1:
            return r0 + (r1 - r0) * (mph - s0) / (s1 - s0)
    return 100.0


def gap_analysis(vehicle_seconds, window_seconds: float, critical_gap: float,
                 step: float = 0.2) -> dict:
    """How long a pedestrian arriving at a random moment waits for a gap of `critical_gap` seconds.

    `vehicle_seconds` are the times (seconds from window start) at which vehicles pass the
    crosswalk. A pedestrian arriving at time a can go if no vehicle arrives within the next
    `critical_gap` seconds; otherwise they wait for the start of the next gap that long.
    Arrivals whose answer depends on traffic after the recording ends are excluded.
    """
    v = np.sort(np.asarray(vehicle_seconds, float))
    v = v[(v >= 0) & (v <= window_seconds)]
    bounds = np.concatenate([[0.0], v, [window_seconds]])
    lengths = np.diff(bounds)
    qualifies = lengths >= critical_gap
    # next_q[i] = index of the first qualifying gap after gap i (or -1)
    next_q = np.full(len(lengths), -1)
    nxt = -1
    for i in range(len(lengths) - 1, -1, -1):
        next_q[i] = nxt
        if qualifies[i]:
            nxt = i

    arrivals = np.arange(0.0, window_seconds, step)
    gap_idx = np.clip(np.searchsorted(bounds, arrivals, side="right") - 1, 0, len(lengths) - 1)
    room = bounds[gap_idx + 1] - arrivals
    delay = np.where(room >= critical_gap, 0.0, np.nan)
    waiting = np.isnan(delay)
    j = next_q[gap_idx[waiting]]
    delay[np.where(waiting)[0][j >= 0]] = bounds[j[j >= 0]] - arrivals[waiting][j >= 0]
    ok = ~np.isnan(delay)

    n = len(v)
    lam = n / window_seconds if window_seconds else 0.0
    random_delay = ((math.exp(lam * critical_gap) - lam * critical_gap - 1) / lam) if lam else 0.0
    headways = np.diff(v)
    d = delay[ok]
    return {
        "critical_gap_s": round(critical_gap, 1),
        "vehicles": int(n),
        "median_headway_s": round(float(np.median(headways)), 1) if len(headways) else None,
        "pct_gaps_long_enough": round(100.0 * float((headways >= critical_gap).mean()), 1) if len(headways) else None,
        "avg_wait_s": round(float(d.mean()), 1) if len(d) else None,
        "pct_must_wait": round(100.0 * float((d > 0).mean()), 1) if len(d) else None,
        "p90_wait_s": round(float(np.percentile(d, 90)), 1) if len(d) else None,
        "max_wait_s": round(float(d.max()), 1) if len(d) else None,
        "los": ped_los(float(d.mean())) if len(d) else None,
        "random_traffic_avg_wait_s": round(random_delay, 1),
        "_delays": d, "_arrivals": arrivals[ok],
    }


def stopping_sight_distance_ft(mph: float, grade_pct: float = 0.0, reaction_s: float = 2.5,
                               decel_fps2: float = 11.2) -> float:
    """AASHTO Green Book stopping sight distance (feet).

    Brake reaction distance plus braking distance. grade_pct is positive uphill and
    negative downhill (a downhill approach needs more distance).
    """
    reaction = 1.47 * mph * reaction_s
    braking = mph ** 2 / (30 * (decel_fps2 / 32.2 + grade_pct / 100))
    return reaction + braking


def design_stopping_sight_distance_ft(mph: float, grade_pct: float = 0.0) -> int:
    """Rounded up to the next 5 ft, as in the Green Book tables."""
    return int(math.ceil(stopping_sight_distance_ft(mph, grade_pct) / 5) * 5)
