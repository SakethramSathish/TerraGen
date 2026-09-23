"""
core/safety.py
==============
**Active Safety Guardian** - the always-on visual indicator layer for the operator.

Two interlock families are monitored:

1. **Seatbelt Status** - latched / unlatched, evaluated against machine motion. A belt
   released while tramming is a stop-work condition; a belt released while parked is merely
   informational.
2. **Proximity Radar** - a three-zone Cat-style detection hierarchy (STOP / WARNING /
   CAUTION) derived from the ``proximity_m`` channel.

Fail-safe philosophy
--------------------
When a sensor is missing, out of range or otherwise unusable the state resolves to
``UNKNOWN`` - **never** to ``OK``. :func:`safety_snapshot` ranks ``UNKNOWN`` above
``WARNING`` when folding the overall level, so a blind radar is treated as an occupied
swing radius rather than as a clear one. This is exactly the behaviour asserted by the
boundary tests (``proximity_m = -1.0`` must not raise and must not report "clear").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from config import THRESHOLDS, Thresholds
from core.telemetry import to_float_frame

#: Indicator levels used by the UI badges (mapped to colours in ``app.py``).
SAFETY_LEVELS: tuple[str, ...] = ("OK", "CAUTION", "WARNING", "UNKNOWN", "CRITICAL")

#: Ranking used to fold several indicators into one overall level.
#: ``UNKNOWN`` deliberately outranks ``WARNING`` (fail-safe).
_LEVEL_RANK: dict[str, float] = {
    "OK": 0.0,
    "CAUTION": 1.0,
    "WARNING": 2.0,
    "UNKNOWN": 2.5,
    "CRITICAL": 3.0,
}

ZONE_STOP: str = "STOP"
ZONE_WARNING: str = "WARNING"
ZONE_CAUTION: str = "CAUTION"
ZONE_CLEAR: str = "CLEAR"
ZONE_UNKNOWN: str = "UNKNOWN"


# --------------------------------------------------------------------------------------
# Indicator states
# --------------------------------------------------------------------------------------
@dataclass
class SeatbeltState:
    """
    Seatbelt interlock indicator state.

    Two views are kept deliberately separate, because an operator needs both:

    * **now** - ``current_latched`` / ``current_level``: the latest sample, i.e. what the
      interlock is reporting at this instant;
    * **window** - ``level``: the worst case across the analysed period, i.e. whether the
      belt was *ever* off while tramming.
    """

    level: str = "UNKNOWN"
    latched: bool | None = None
    current_level: str = "UNKNOWN"
    current_latched: bool | None = None
    current_moving: bool = False
    message: str = "No seatbelt data available"
    current_message: str = "No seatbelt data available"
    unlatched_moving_seconds: float = 0.0
    unlatched_seconds: float = 0.0
    compliance_pct: float = 100.0

    @property
    def needs_attention(self) -> bool:
        """True when the operator must act before continuing work."""
        return self.level in {"WARNING", "CRITICAL", "UNKNOWN"}


@dataclass
class ProximityState:
    """Proximity-radar indicator state."""

    level: str = "UNKNOWN"
    zone: str = ZONE_UNKNOWN
    nearest_m: float | None = None
    current_m: float | None = None
    current_zone: str = ZONE_UNKNOWN
    current_level: str = "UNKNOWN"
    stop_events: int = 0
    warning_events: int = 0
    message: str = "Proximity radar unavailable"
    sensor_faults: int = 0

    @property
    def needs_attention(self) -> bool:
        """True when the zone demands a reduced swing or a full stop."""
        return self.zone in {ZONE_STOP, ZONE_WARNING, ZONE_UNKNOWN}


@dataclass
class SafetySnapshot:
    """Composite safety picture used to drive the Safety Guardian tab."""

    overall_level: str = "UNKNOWN"
    safety_score: int = 0
    seatbelt: SeatbeltState = field(default_factory=SeatbeltState)
    proximity: ProximityState = field(default_factory=ProximityState)
    machine_id: str = "UNKNOWN"
    window_start: pd.Timestamp | None = None
    window_end: pd.Timestamp | None = None
    samples: int = 0
    blockers: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)

    @property
    def all_clear(self) -> bool:
        """True only when every indicator is ``OK``."""
        return self.overall_level == "OK"

    def headline(self) -> str:
        """Short status line for the operator HUD."""
        if self.overall_level == "OK":
            return "All safety interlocks nominal"
        if not self.samples:
            return "No telemetry - safety state unknown"
        return "; ".join(self.blockers) if self.blockers else f"Safety level {self.overall_level}"


# --------------------------------------------------------------------------------------
# Evaluators
# --------------------------------------------------------------------------------------
def _movement_series(work: pd.DataFrame) -> pd.Series:
    """Ground-speed series with sensible fallbacks for a malformed frame."""
    if "ground_speed_kph" in work.columns:
        return work["ground_speed_kph"].astype("float64")
    return pd.Series(np.zeros(len(work)), index=work.index, dtype="float64")


def evaluate_seatbelt(
    frame: pd.DataFrame,
    thresholds: Thresholds = THRESHOLDS,
) -> SeatbeltState:
    """
    Evaluate seatbelt compliance over the window.

    * unlatched **while moving** (speed > ``seatbelt_release_speed_kph``) -> ``CRITICAL``
    * unlatched while stationary                                  -> ``CAUTION``
    * latched throughout                                          -> ``OK``
    * channel missing / unreadable                                -> ``UNKNOWN``
    """
    state = SeatbeltState()
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        state.message = "No telemetry - seatbelt interlock state unknown"
        return state
    if "seatbelt_latched" not in frame.columns:
        state.message = "Seatbelt channel not reported by the CAN gateway"
        return state

    latched = frame["seatbelt_latched"].astype("boolean")
    usable = latched.notna()
    if not bool(usable.any()):
        state.message = "Seatbelt channel unreadable (all samples invalid)"
        return state

    work = to_float_frame(frame)
    speed = _movement_series(work)
    interval = float(thresholds.sample_interval_s)
    is_latched = latched.fillna(True).astype(bool)
    moving = (speed > thresholds.seatbelt_release_speed_kph).fillna(False).astype(bool)

    unlatched = ~is_latched
    state.unlatched_seconds = round(float(unlatched.sum()) * interval, 1)
    state.unlatched_moving_seconds = round(float((unlatched & moving).sum()) * interval, 1)
    state.compliance_pct = round(float(is_latched.mean() * 100.0), 1)

    # --- "now" view: the most recent usable sample ---------------------------------
    last_position = is_latched.index[-1]
    for position in reversed(list(is_latched.index)):
        if bool(usable.loc[position]):
            last_position = position
            break
    state.latched = bool(is_latched.loc[last_position])
    state.current_latched = state.latched
    state.current_moving = bool(moving.loc[last_position])
    if state.current_latched:
        state.current_level = "OK"
        state.current_message = "Seatbelt latched - interlock satisfied."
    elif state.current_moving:
        state.current_level = "CRITICAL"
        state.current_message = (
            "Seatbelt is UNLATCHED while the machine is moving. Stop in a safe area, latch the "
            "belt and confirm the indicator clears before resuming."
        )
    else:
        state.current_level = "CAUTION"
        state.current_message = "Seatbelt released while stationary - latch before tramming."

    if bool((unlatched & moving).any()):
        state.level = "CRITICAL"
        state.message = (
            f"SEATBELT UNLATCHED WHILE MOVING - {state.unlatched_moving_seconds:.0f} s logged. "
            "Stop the machine, latch the belt, then resume."
        )
    elif bool(unlatched.any()):
        state.level = "CAUTION"
        state.message = (
            f"Seatbelt released for {state.unlatched_seconds:.0f} s while stationary - "
            "latch before tramming."
        )
    else:
        state.level = "OK"
        state.message = f"Seatbelt latched for the full window ({state.compliance_pct:.0f} % compliance)"
    return state


def evaluate_proximity(
    frame: pd.DataFrame,
    thresholds: Thresholds = THRESHOLDS,
) -> ProximityState:
    """
    Classify the proximity channel into radar zones.

    Zones: ``STOP`` (< 5 m), ``WARNING`` (< 10 m), ``CAUTION`` (< 15 m), ``CLEAR``.
    Negative or unparsable readings are rejected upstream by the sanitiser; if any survive
    as ``NaN`` the state becomes ``UNKNOWN`` (fail-safe, never "clear").
    """
    state = ProximityState()
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        state.message = "No telemetry - proximity radar unknown, treat swing radius as occupied"
        return state
    if "proximity_m" not in frame.columns:
        state.message = "Proximity channel not reported by the CAN gateway"
        return state

    values = pd.to_numeric(frame["proximity_m"], errors="coerce").astype("float64")
    faults = int(values.isna().sum())
    state.sensor_faults = faults
    if state.sensor_faults == len(values):
        state.message = "Proximity radar fault - treat swing radius as occupied, use a spotter"
        return state

    usable = values.dropna()
    state.nearest_m = round(float(usable.min()), 2)
    if len(usable):
        state.current_m = round(float(usable.iloc[-1]), 2)
        state.current_zone, state.current_level = proximity_zone_from_reading(
            state.current_m, thresholds
        )
    state.stop_events = int((usable < thresholds.proximity_stop_m).sum())
    state.warning_events = int((usable < thresholds.proximity_warning_m).sum())

    if state.nearest_m < thresholds.proximity_stop_m:
        state.zone, state.level = ZONE_STOP, "CRITICAL"
        state.message = (
            f"STOP - object at {state.nearest_m:.1f} m (inside the {thresholds.proximity_stop_m:.0f} m "
            "stop zone). Stop the swing/travel, sound the horn and wait for the spotter."
        )
    elif state.nearest_m < thresholds.proximity_warning_m:
        state.zone, state.level = ZONE_WARNING, "WARNING"
        state.message = (
            f"Object at {state.nearest_m:.1f} m - inside the {thresholds.proximity_warning_m:.0f} m "
            "warning zone. Slow the swing, keep the load low and re-check mirrors/camera."
        )
    elif state.nearest_m < thresholds.proximity_caution_m:
        state.zone, state.level = ZONE_CAUTION, "CAUTION"
        state.message = f"Object at {state.nearest_m:.1f} m - caution zone, maintain awareness."
    else:
        state.zone, state.level = ZONE_CLEAR, "OK"
        state.message = f"Radar clear - nearest object {state.nearest_m:.1f} m."

    if faults:
        state.level = "WARNING" if state.level == "OK" else state.level
        state.message += f" ({faults} invalid radar sample(s) ignored)"
    return state


def safety_score(
    seatbelt: SeatbeltState,
    proximity: ProximityState,
    integrity_ratio: float = 1.0,
) -> int:
    """
    Blend the indicators + data integrity into a 0-100 operator score.

    Weights: seatbelt 40 · proximity 40 · data integrity 20.
    """
    belt_points = {"OK": 40.0, "CAUTION": 20.0, "WARNING": 10.0, "CRITICAL": 0.0, "UNKNOWN": 0.0}
    prox_points = {
        "OK": 40.0,
        "CAUTION": 26.0,
        "WARNING": 14.0,
        "CRITICAL": 0.0,
        "UNKNOWN": 0.0,
    }
    ratio = float(min(1.0, max(0.0, integrity_ratio)))
    total = belt_points.get(seatbelt.level, 0.0) + prox_points.get(proximity.level, 0.0) + 20.0 * ratio
    return int(round(min(100.0, max(0.0, total))))


def safety_snapshot(
    frame: pd.DataFrame,
    thresholds: Thresholds = THRESHOLDS,
) -> SafetySnapshot:
    """
    Build the full :class:`SafetySnapshot` for the Safety Guardian tab.

    Total function: never raises for empty, partial or corrupted frames.
    """
    snapshot = SafetySnapshot()
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        snapshot.overall_level = "UNKNOWN"
        snapshot.blockers = ["No telemetry available - safety state unknown"]
        snapshot.safety_score = 0
        return snapshot

    snapshot.samples = int(len(frame))
    if "machine_id" in frame.columns and len(frame):
        snapshot.machine_id = str(frame["machine_id"].iloc[-1])[:32]

    timestamps = pd.to_datetime(frame.get("timestamp", pd.Series(dtype="datetime64[ns]")), errors="coerce")
    if not timestamps.dropna().empty:
        snapshot.window_start = timestamps.dropna().iloc[0]
        snapshot.window_end = timestamps.dropna().iloc[-1]

    snapshot.seatbelt = evaluate_seatbelt(frame, thresholds)
    snapshot.proximity = evaluate_proximity(frame, thresholds)

    integrity = 1.0
    if "sensor_fault" in frame.columns:
        bad = int(frame["sensor_fault"].fillna(False).astype(bool).sum())
        integrity = 1.0 - (bad / max(1, len(frame)))

    snapshot.safety_score = safety_score(snapshot.seatbelt, snapshot.proximity, integrity)

    level = "OK"
    for candidate in (
        snapshot.seatbelt.level,
        snapshot.seatbelt.current_level,
        snapshot.proximity.level,
        snapshot.proximity.current_level,
    ):
        if _LEVEL_RANK.get(candidate, 3.0) > _LEVEL_RANK.get(level, 0.0):
            level = candidate
    snapshot.overall_level = level

    blockers: list[str] = []
    for indicator_level, message in (
        (snapshot.seatbelt.level, snapshot.seatbelt.message),
        (snapshot.proximity.level, snapshot.proximity.message),
    ):
        if indicator_level != "OK":
            blockers.append(message)
    if integrity < 1.0:
        blockers.append(f"{round((1 - integrity) * 100)} % of samples had invalid sensor fields")
    snapshot.blockers = blockers
    return snapshot


def proximity_zone_from_reading(
    reading: Any,
    thresholds: Thresholds = THRESHOLDS,
) -> tuple[str, str]:
    """
    Classify a *single* radar reading (used by the live edge simulator).

    Returns ``(zone, level)``; non-finite readings return ``("UNKNOWN", "UNKNOWN")``.
    """
    try:
        value = float(reading)
    except (TypeError, ValueError):
        return ZONE_UNKNOWN, "UNKNOWN"
    if not math.isfinite(value):
        return ZONE_UNKNOWN, "UNKNOWN"
    if value < thresholds.proximity_stop_m:
        return ZONE_STOP, "CRITICAL"
    if value < thresholds.proximity_warning_m:
        return ZONE_WARNING, "WARNING"
    if value < thresholds.proximity_caution_m:
        return ZONE_CAUTION, "CAUTION"
    return ZONE_CLEAR, "OK"


__all__ = [
    "SAFETY_LEVELS",
    "ZONE_STOP",
    "ZONE_WARNING",
    "ZONE_CAUTION",
    "ZONE_CLEAR",
    "ZONE_UNKNOWN",
    "SeatbeltState",
    "ProximityState",
    "SafetySnapshot",
    "evaluate_seatbelt",
    "evaluate_proximity",
    "safety_score",
    "safety_snapshot",
    "proximity_zone_from_reading",
]
