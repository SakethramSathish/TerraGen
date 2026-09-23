"""
core/dashboard.py
=================
**Daily Task Dashboard** domain logic: shift plan, milestones, haul-tonnage progress and
machine status.

Everything here is a pure function of ``(plan, kpis, now)`` so the dashboard is:

* **testable** - ``now`` is always injected, never read from the wall clock inside logic;
* **offline** - no scheduler, no database, no external service;
* **idempotent** - re-rendering the page cannot mutate the shift record.

Shift model
-----------
A shift is a fixed window (``06:00-18:00`` day, ``18:00-06:00`` night) with a tonnage
target and a set of milestone bands at 10 / 25 / 50 / 75 / 100 % of the plan. The dashboard
answers the three questions an operator actually has at 09:00:

1. *How much have I moved?*        -> ``tons_hauled``, ``cycles``, ``progress_pct``
2. *Am I ahead or behind?*         -> ``required_rate_tph`` vs ``actual_rate_tph``, ``forecast``
3. *What is the next milestone?*   -> ``next_milestone``, ``eta_to_target``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from config import (
    DEFAULT_SHIFT_HOURS,
    DEFAULT_SHIFT_TARGET_CYCLES,
    DEFAULT_SHIFT_TARGET_TONS,
    THRESHOLDS,
    Thresholds,
)
from core.telemetry import KpiSummary, compute_kpis, machine_status

#: Milestone bands as (fraction of plan, label, coach note).
MILESTONE_BANDS: tuple[tuple[float, str, str], ...] = (
    (0.10, "Warm-up & first 10 %", "Confirm pre-start checks, fluid levels and tyre/undercarriage."),
    (0.25, "Quarter plan", "First fuel/cycle sanity check - compare L/t against yesterday."),
    (0.50, "Half plan (mid-shift)", "Mid-shift walk-around, bucket teeth and track tension check."),
    (0.75, "Three-quarter plan", "Check tank level; plan the refuel window to avoid the last hour."),
    (1.00, "Shift target met", "Log the shift, capture fuel burn and any open fault codes."),
)

FORECAST_ON_TRACK = "ON TRACK"
FORECAST_BEHIND = "BEHIND PLAN"
FORECAST_AHEAD = "AHEAD OF PLAN"
FORECAST_UNKNOWN = "NO DATA"


# --------------------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------------------
@dataclass
class Milestone:
    """One target band within the shift plan."""

    label: str
    target_tons: float
    fraction: float
    note: str = ""
    target_cycles: int = 0
    achieved: bool = False
    achieved_at: pd.Timestamp | None = None
    at_risk: bool = False

    @property
    def status(self) -> str:
        """``DONE`` | ``AT RISK`` | ``PENDING``."""
        if self.achieved:
            return "DONE"
        return "AT RISK" if self.at_risk else "PENDING"


@dataclass
class ShiftPlan:
    """Immutable shift definition (target, window and milestone schedule)."""

    shift_name: str = "DAY SHIFT"
    start: datetime = field(default_factory=lambda: datetime.now().replace(hour=6, minute=0, second=0, microsecond=0))
    hours: float = DEFAULT_SHIFT_HOURS
    target_tons: float = DEFAULT_SHIFT_TARGET_TONS
    target_cycles: int = DEFAULT_SHIFT_TARGET_CYCLES
    machine_id: str = "CAT-320-EXC-014"
    operator: str = "Operator"
    milestones: list[Milestone] = field(default_factory=list)

    @property
    def end(self) -> datetime:
        """Shift end timestamp."""
        return self.start + timedelta(hours=float(self.hours))

    def duration_hours(self) -> float:
        """Planned shift duration in hours."""
        return max(0.01, float(self.hours))


@dataclass
class DashboardState:
    """Everything the Daily Task Dashboard renders."""

    plan: ShiftPlan
    now: datetime
    tons_hauled: float = 0.0
    cycles: int = 0
    progress_pct: float = 0.0
    tons_remaining: float = 0.0
    elapsed_hours: float = 0.0
    remaining_hours: float = 0.0
    required_rate_tph: float = 0.0
    actual_rate_tph: float = 0.0
    projected_tons: float = 0.0
    forecast: str = FORECAST_UNKNOWN
    forecast_delta_tons: float = 0.0
    machine_status: str = "NO_DATA"
    eta_to_target: datetime | None = None
    milestones: list[Milestone] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)

    @property
    def next_milestone(self) -> Milestone | None:
        """The first milestone that has not been reached yet."""
        for milestone in self.milestones:
            if not milestone.achieved:
                return milestone
        return None

    @property
    def target_met(self) -> bool:
        """True once the tonnage target has been reached."""
        return self.tons_hauled >= self.plan.target_tons > 0


# --------------------------------------------------------------------------------------
# Plan construction
# --------------------------------------------------------------------------------------
def build_shift_plan(
    now: datetime | None = None,
    *,
    shift: str = "AUTO",
    target_tons: float = DEFAULT_SHIFT_TARGET_TONS,
    target_cycles: int = DEFAULT_SHIFT_TARGET_CYCLES,
    shift_hours: float = DEFAULT_SHIFT_HOURS,
    machine_id: str = "CAT-320-EXC-014",
    operator: str = "Operator",
) -> ShiftPlan:
    """
    Build a :class:`ShiftPlan` for the requested window.

    ``shift="AUTO"`` picks the shift that *contains* ``now`` (day 06:00-18:00, night
    18:00-06:00, where a night shift starting yesterday at 18:00 still owns early-morning
    hours). Targets are clamped to sane, positive values so a fat-fingered sidebar entry
    can never divide-by-zero the dashboard.
    """
    now = now or datetime.now()
    if str(shift).upper().startswith("DAY"):
        start = now.replace(hour=6, minute=0, second=0, microsecond=0)
        name = "DAY SHIFT"
    elif str(shift).upper().startswith("NIGHT"):
        candidate = now.replace(hour=18, minute=0, second=0, microsecond=0)
        start = candidate if now >= candidate else candidate - timedelta(days=1)
        name = "NIGHT SHIFT"
    else:
        if 6 <= now.hour < 18:
            start = now.replace(hour=6, minute=0, second=0, microsecond=0)
            name = "DAY SHIFT"
        else:
            start = (now - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0) \
                if now.hour < 6 else now.replace(hour=18, minute=0, second=0, microsecond=0)
            name = "NIGHT SHIFT"

    tons = max(1.0, float(target_tons))
    cycles = max(1, int(target_cycles))
    hours = max(0.25, float(shift_hours))

    plan = ShiftPlan(
        shift_name=name,
        start=start,
        hours=hours,
        target_tons=tons,
        target_cycles=cycles,
        machine_id=machine_id,
        operator=operator,
    )
    plan.milestones = [
        Milestone(
            label=label,
            target_tons=round(tons * fraction, 2),
            fraction=fraction,
            note=note,
            target_cycles=int(round(cycles * fraction)),
        )
        for fraction, label, note in MILESTONE_BANDS
    ]
    return plan


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def evaluate_shift(
    plan: ShiftPlan,
    kpis: KpiSummary | None = None,
    *,
    now: datetime | None = None,
    thresholds: Thresholds = THRESHOLDS,
) -> DashboardState:
    """
    Combine the plan with the shift-to-date KPIs into a renderable state.

    Guards every division, survives ``kpis=None`` and a zero target, and never returns a
    negative duration (an operator forwarding the machine clock cannot break the page).
    """
    now = now or datetime.now()
    kpis = kpis or KpiSummary()

    state = DashboardState(plan=plan, now=now)
    state.tons_hauled = round(max(0.0, float(kpis.tons_hauled)), 2)
    state.cycles = max(0, int(kpis.cycles))

    target = plan.target_tons if plan.target_tons > 0 else 0.0
    state.progress_pct = round(min(999.0, (state.tons_hauled / target * 100.0) if target else 0.0), 1)
    state.tons_remaining = round(max(0.0, target - state.tons_hauled), 2)

    elapsed = (now - plan.start).total_seconds() / 3600.0
    state.elapsed_hours = round(max(0.0, min(elapsed, plan.duration_hours())), 2)
    state.remaining_hours = round(max(0.0, plan.duration_hours() - state.elapsed_hours), 2)

    state.required_rate_tph = (
        round(state.tons_remaining / state.remaining_hours, 2) if state.remaining_hours > 0 else 0.0
    )
    state.actual_rate_tph = (
        round(state.tons_hauled / state.elapsed_hours, 2) if state.elapsed_hours > 0 else 0.0
    )
    state.projected_tons = round(state.actual_rate_tph * plan.duration_hours(), 2)
    state.forecast_delta_tons = round(state.projected_tons - target, 2)

    tolerance = thresholds.behind_plan_tolerance_pct / 100.0
    if kpis.rows == 0 or state.elapsed_hours <= 0:
        state.forecast = FORECAST_UNKNOWN
    elif state.projected_tons >= target * (1.0 + tolerance):
        state.forecast = FORECAST_AHEAD
    elif state.projected_tons < target * (1.0 - tolerance):
        state.forecast = FORECAST_BEHIND
    else:
        state.forecast = FORECAST_ON_TRACK

    if state.actual_rate_tph > 0 and state.tons_remaining > 0:
        eta_hours = state.tons_remaining / state.actual_rate_tph
        state.eta_to_target = now + timedelta(hours=eta_hours)

    # --- milestone evaluation ------------------------------------------------------
    for milestone in plan.milestones:
        milestone.achieved = state.tons_hauled >= milestone.target_tons
        milestone.achieved_at = None
        elapsed_fraction = state.elapsed_hours / plan.duration_hours() if plan.duration_hours() else 0.0
        milestone.at_risk = (
            not milestone.achieved
            and elapsed_fraction > milestone.fraction + tolerance
        )
    state.milestones = list(plan.milestones)

    # --- operator coaching (static strings; no user data interpolated) -------------
    lessons: list[str] = []
    if kpis.idle_pct >= 35.0:
        lessons.append(
            f"Idle time is {kpis.idle_pct:.0f} % of the shift - the biggest single fuel lever today."
        )
    if kpis.fuel_per_ton_l and kpis.fuel_per_ton_l > thresholds.fuel_per_ton_l_max:
        lessons.append(
            f"Fuel burn is {kpis.fuel_per_ton_l:.3f} L/t (limit "
            f"{thresholds.fuel_per_ton_l_max:.2f}) - trim bucket fill and cut swing idle."
        )
    if state.forecast == FORECAST_BEHIND:
        lessons.append(
            f"Projection is {abs(state.forecast_delta_tons):.0f} t short of plan - "
            "reduce truck-spotting time between cycles."
        )
    if kpis.critical_events:
        lessons.append(
            f"{kpis.critical_events} critical safety sample(s) logged - review the Safety Guardian tab."
        )
    state.lessons = lessons
    return state


def shift_rollup(
    frame: pd.DataFrame,
    plan: ShiftPlan,
    *,
    now: datetime | None = None,
    thresholds: Thresholds = THRESHOLDS,
) -> DashboardState:
    """Convenience wrapper: ``frame -> KPIs -> DashboardState`` (used by ``app.py``)."""
    kpis = compute_kpis(frame, thresholds)
    state = evaluate_shift(plan, kpis, now=now, thresholds=thresholds)
    state.machine_status = machine_status(frame, thresholds)
    return state


def milestone_table(state: DashboardState) -> list[dict[str, object]]:
    """Flatten the milestone list into rows for ``st.dataframe``."""
    remaining = state.plan.target_tons - state.tons_hauled
    rows: list[dict[str, object]] = []
    for milestone in state.milestones:
        rows.append(
            {
                "Milestone": milestone.label,
                "Status": milestone.status,
                "Target (t)": f"{milestone.target_tons:,.0f}",
                "Achieved (t)": f"{min(state.tons_hauled, milestone.target_tons):,.0f}",
                "Gap (t)": f"{max(0.0, milestone.target_tons - state.tons_hauled):,.0f}",
                "Cycles (plan)": milestone.target_cycles,
            }
        )
    rows.append(
        {
            "Milestone": "SHIFT TOTAL",
            "Status": "DONE" if state.target_met else state.forecast,
            "Target (t)": f"{state.plan.target_tons:,.0f}",
            "Achieved (t)": f"{state.tons_hauled:,.0f}",
            "Gap (t)": f"{max(0.0, remaining):,.0f}",
            "Cycles (plan)": state.plan.target_cycles,
        }
    )
    return rows


def shift_time_series(
    frame: pd.DataFrame,
    plan: ShiftPlan,
    thresholds: Thresholds = THRESHOLDS,
) -> pd.DataFrame:
    """
    Build a cumulative-tons progress curve for the dashboard chart.

    Returns a frame with ``timestamp``, ``cumulative_tons``, ``plan_tons`` and
    ``gap_tons`` - the plan line is derived from the shift clock, so the operator sees
    the gap to plan at every instant.
    """
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=["timestamp", "cumulative_tons", "plan_tons", "gap_tons"])

    from core.telemetry import detect_haul_cycles  # local import avoids a cycle at import time

    cycles = detect_haul_cycles(frame, thresholds)
    if not cycles:
        return pd.DataFrame(columns=["timestamp", "cumulative_tons", "plan_tons", "gap_tons"])

    record = pd.DataFrame(
        {
            "timestamp": [c.end for c in cycles],
            "cycle_tons": [c.peak_payload_tons for c in cycles],
        }
    ).sort_values("timestamp")
    record["cumulative_tons"] = record["cycle_tons"].cumsum().round(2)

    duration_h = plan.duration_hours()
    start = pd.Timestamp(plan.start)
    record["plan_tons"] = [
        round(
            plan.target_tons
            * min(1.0, max(0.0, (ts - start).total_seconds() / 3600.0) / duration_h),
            2,
        )
        for ts in record["timestamp"]
    ]
    record["gap_tons"] = (record["cumulative_tons"] - record["plan_tons"]).round(2)
    return record.reset_index(drop=True)


__all__ = [
    "MILESTONE_BANDS",
    "FORECAST_ON_TRACK",
    "FORECAST_BEHIND",
    "FORECAST_AHEAD",
    "FORECAST_UNKNOWN",
    "Milestone",
    "ShiftPlan",
    "DashboardState",
    "build_shift_plan",
    "evaluate_shift",
    "shift_rollup",
    "milestone_table",
    "shift_time_series",
]
