"""
tests/test_dashboard_safety.py
==============================
Unit tests for the **Daily Task Dashboard** domain logic, the **Active Safety Guardian**
indicators and the presentation helpers.

All time-dependent logic is exercised with an injected ``now``, so the suite is stable at
02:00 during a night shift or during a leap-year DST change - and the tests document exactly
what the dashboard shows in each state.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

import config
from core import dashboard, formatting as fmt, safety, telemetry
from tests.conftest import build_frame

DAY_START = datetime(2026, 9, 23, 6, 0, 0)


# ======================================================================================
# Shift plan
# ======================================================================================
class TestShiftPlan:
    """Plan construction: windows, targets, milestones and sane clamping."""

    def test_day_shift_window(self) -> None:
        plan = dashboard.build_shift_plan(datetime(2026, 9, 23, 9, 30), shift="DAY", shift_hours=12)
        assert plan.start == DAY_START
        assert plan.end == datetime(2026, 9, 23, 18, 0)
        assert plan.shift_name == "DAY SHIFT"

    def test_night_shift_owns_the_early_morning(self) -> None:
        """At 02:00 the operator is still on the night shift that started yesterday at 18:00."""
        plan = dashboard.build_shift_plan(datetime(2026, 9, 23, 2, 0), shift="AUTO")
        assert plan.shift_name == "NIGHT SHIFT"
        assert plan.start == datetime(2026, 9, 22, 18, 0)

    def test_auto_selects_the_containing_shift(self) -> None:
        assert dashboard.build_shift_plan(datetime(2026, 9, 23, 9, 0), shift="AUTO").shift_name == "DAY SHIFT"
        assert dashboard.build_shift_plan(datetime(2026, 9, 23, 20, 0), shift="AUTO").shift_name == "NIGHT SHIFT"

    def test_targets_are_clamped_to_positive_values(self) -> None:
        plan = dashboard.build_shift_plan(target_tons=0, target_cycles=0, shift_hours=0)
        assert plan.target_tons >= 1.0
        assert plan.target_cycles >= 1
        assert plan.duration_hours() > 0

    def test_milestones_span_the_plan(self) -> None:
        plan = dashboard.build_shift_plan(target_tons=2_000, target_cycles=200)
        fractions = [milestone.fraction for milestone in plan.milestones]
        assert fractions == sorted(fractions)
        assert fractions[-1] == pytest.approx(1.0)
        assert plan.milestones[-1].target_tons == pytest.approx(2_000.0)
        assert plan.milestones[0].target_tons == pytest.approx(200.0)
        assert all(milestone.note for milestone in plan.milestones)  # coaching text present


# ======================================================================================
# Dashboard evaluation
# ======================================================================================
class TestDashboardEvaluation:
    """Progress, forecast and milestone states."""

    def _plan(self, target_tons: float = 1_200.0) -> dashboard.ShiftPlan:
        return dashboard.build_shift_plan(DAY_START, shift="DAY", target_tons=target_tons,
                                          target_cycles=120, shift_hours=12)

    def test_progress_and_remaining(self) -> None:
        kpis = telemetry.KpiSummary(rows=500, tons_hauled=300.0, cycles=30, window_hours=3.0)
        state = dashboard.evaluate_shift(self._plan(), kpis, now=DAY_START + timedelta(hours=3))
        assert state.progress_pct == pytest.approx(25.0, abs=0.1)
        assert state.tons_remaining == pytest.approx(900.0, abs=0.1)
        assert state.elapsed_hours == pytest.approx(3.0, abs=0.01)
        assert state.remaining_hours == pytest.approx(9.0, abs=0.01)
        assert state.actual_rate_tph == pytest.approx(100.0, abs=0.1)
        assert state.required_rate_tph == pytest.approx(100.0, abs=0.1)

    def test_behind_plan_forecast(self) -> None:
        kpis = telemetry.KpiSummary(rows=500, tons_hauled=120.0, cycles=12, window_hours=8.0)
        state = dashboard.evaluate_shift(self._plan(), kpis, now=DAY_START + timedelta(hours=8))
        assert state.forecast == dashboard.FORECAST_BEHIND
        assert state.forecast_delta_tons < 0
        assert state.projected_tons == pytest.approx(180.0, abs=1.0)

    def test_ahead_of_plan_forecast(self) -> None:
        kpis = telemetry.KpiSummary(rows=500, tons_hauled=600.0, cycles=60, window_hours=4.0)
        state = dashboard.evaluate_shift(self._plan(), kpis, now=DAY_START + timedelta(hours=4))
        assert state.forecast == dashboard.FORECAST_AHEAD

    def test_on_track_within_tolerance(self) -> None:
        # 100 t/h over 12 h = 1 200 t exactly on plan.
        kpis = telemetry.KpiSummary(rows=500, tons_hauled=400.0, cycles=40, window_hours=4.0)
        state = dashboard.evaluate_shift(self._plan(), kpis, now=DAY_START + timedelta(hours=4))
        assert state.forecast == dashboard.FORECAST_ON_TRACK

    def test_no_data_yields_unknown_forecast(self) -> None:
        state = dashboard.evaluate_shift(self._plan(), telemetry.KpiSummary(), now=DAY_START)
        assert state.forecast == dashboard.FORECAST_UNKNOWN
        assert state.tons_hauled == 0.0
        assert state.actual_rate_tph == 0.0

    def test_milestone_states_and_at_risk_detection(self) -> None:
        kpis = telemetry.KpiSummary(rows=500, tons_hauled=120.0, cycles=12, window_hours=8.0)
        state = dashboard.evaluate_shift(self._plan(), kpis, now=DAY_START + timedelta(hours=8))
        by_fraction = {milestone.fraction: milestone for milestone in state.milestones}
        assert by_fraction[0.10].achieved is True
        assert by_fraction[0.10].status == "DONE"
        assert by_fraction[0.25].at_risk is True  # 8 h elapsed, only 10 % of plan moved
        assert by_fraction[0.25].status == "AT RISK"
        assert by_fraction[1.00].status in {"PENDING", "AT RISK"}
        assert state.next_milestone is not None
        assert state.next_milestone.fraction == 0.25

    def test_eta_is_projected_from_the_actual_rate(self) -> None:
        kpis = telemetry.KpiSummary(rows=500, tons_hauled=300.0, cycles=30, window_hours=3.0)
        now = DAY_START + timedelta(hours=3)
        state = dashboard.evaluate_shift(self._plan(), kpis, now=now)
        assert state.eta_to_target is not None
        assert state.eta_to_target > now
        # 900 t remaining at 100 t/h -> 9 h from "now"
        assert abs((state.eta_to_target - now) - timedelta(hours=9)) < timedelta(minutes=1)

    def test_target_met_flag(self) -> None:
        kpis = telemetry.KpiSummary(rows=500, tons_hauled=1_300.0, cycles=130, window_hours=11.0)
        state = dashboard.evaluate_shift(self._plan(), kpis, now=DAY_START + timedelta(hours=11))
        assert state.target_met is True
        assert state.tons_remaining == 0.0

    def test_elapsed_is_clamped_to_the_shift(self) -> None:
        kpis = telemetry.KpiSummary(rows=500, tons_hauled=1_300.0, cycles=130, window_hours=14.0)
        state = dashboard.evaluate_shift(self._plan(), kpis, now=DAY_START + timedelta(hours=30))
        assert state.elapsed_hours == pytest.approx(12.0, abs=0.01)
        assert state.remaining_hours == 0.0
        assert state.required_rate_tph == 0.0  # no division by zero

    def test_milestone_table_shape(self) -> None:
        kpis = telemetry.KpiSummary(rows=100, tons_hauled=100.0, cycles=10, window_hours=2.0)
        state = dashboard.evaluate_shift(self._plan(), kpis, now=DAY_START + timedelta(hours=2))
        rows = dashboard.milestone_table(state)
        assert len(rows) == len(state.milestones) + 1
        assert rows[-1]["Milestone"] == "SHIFT TOTAL"
        assert set(rows[0]) == {"Milestone", "Status", "Target (t)", "Achieved (t)", "Gap (t)", "Cycles (plan)"}

    def test_shift_time_series_tracks_the_plan_line(self, analysed: pd.DataFrame) -> None:
        plan = dashboard.build_shift_plan(DAY_START, shift="DAY", target_tons=2_000)
        series = dashboard.shift_time_series(analysed, plan)
        assert not series.empty
        assert series["cumulative_tons"].is_monotonic_increasing
        assert series["plan_tons"].is_monotonic_increasing
        assert (series["plan_tons"] <= 2_000 + 1e-6).all()
        assert series["gap_tons"].iloc[-1] == pytest.approx(
            series["cumulative_tons"].iloc[-1] - series["plan_tons"].iloc[-1], abs=0.02
        )

    def test_shift_time_series_on_empty_input(self, null_frame) -> None:
        series = dashboard.shift_time_series(null_frame, dashboard.build_shift_plan(DAY_START))
        assert series.empty
        assert list(series.columns) == ["timestamp", "cumulative_tons", "plan_tons", "gap_tons"]

    def test_shift_rollup_end_to_end(self, analysed: pd.DataFrame) -> None:
        plan = dashboard.build_shift_plan(DAY_START, shift="DAY")
        state = dashboard.shift_rollup(analysed, plan, now=DAY_START + timedelta(hours=1))
        assert state.machine_status
        assert state.cycles > 0
        assert state.tons_hauled > 0
        assert any("idle" in lesson.lower() for lesson in state.lessons)

    def test_coaching_lessons_are_static_strings(self, analysed: pd.DataFrame) -> None:
        plan = dashboard.build_shift_plan(DAY_START)
        state = dashboard.shift_rollup(analysed, plan, now=DAY_START + timedelta(hours=1))
        for lesson in state.lessons:
            assert "<" not in lesson and ">" not in lesson


# ======================================================================================
# Safety Guardian
# ======================================================================================
class TestSafetyGuardian:
    """Seatbelt interlock and proximity radar indicators."""

    def _sanitise(self, frame: pd.DataFrame) -> pd.DataFrame:
        return telemetry.sanitize_telemetry_frame(frame).frame

    def test_seatbelt_latched_all_window(self) -> None:
        state = safety.evaluate_seatbelt(self._sanitise(build_frame(10)))
        assert state.level == "OK" and state.current_level == "OK"
        assert state.latched is True and state.current_latched is True
        assert state.compliance_pct == 100.0
        assert state.needs_attention is False

    def test_seatbelt_unlatched_while_moving_is_critical(self) -> None:
        frame = build_frame(10, seatbelt_latched=[True] * 8 + [False, False],
                            ground_speed_kph=[3.0] * 8 + [9.0, 9.0])
        state = safety.evaluate_seatbelt(self._sanitise(frame))
        assert state.level == "CRITICAL"
        assert state.unlatched_moving_seconds == pytest.approx(10.0, abs=0.1)  # 2 samples × 5 s
        assert "UNLATCHED WHILE MOVING" in state.message
        assert state.needs_attention is True

    def test_seatbelt_released_while_parked_is_a_caution(self) -> None:
        frame = build_frame(6, seatbelt_latched=[True, True, True, False, False, False],
                            ground_speed_kph=0.0, payload_tons=0.0, rpm=900)
        state = safety.evaluate_seatbelt(self._sanitise(frame))
        assert state.level == "CAUTION"
        assert state.unlatched_moving_seconds == 0.0

    def test_seatbelt_window_verdict_versus_current_state(self) -> None:
        """The belt was off mid-window (while parked) but is latched now: report both."""
        frame = build_frame(
            9,
            seatbelt_latched=[True, True, False, False, True, True, True, True, True],
            ground_speed_kph=[3.0, 3.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        )
        state = safety.evaluate_seatbelt(self._sanitise(frame))
        assert state.level == "CAUTION"          # window worst case
        assert state.current_latched is True     # what the indicator shows now
        assert state.current_level == "OK"

    def test_seatbelt_unknown_when_channel_missing(self, frame_factory) -> None:
        frame = self._sanitise(frame_factory(4).drop(columns=["seatbelt_latched"]))
        state = safety.evaluate_seatbelt(frame)
        assert state.level == "UNKNOWN" and state.current_level == "UNKNOWN"
        assert state.needs_attention is True  # fail-safe, never "OK"

    @pytest.mark.parametrize(
        "distance,expected_zone,expected_level",
        (
            (25.0, safety.ZONE_CLEAR, "OK"),
            (12.0, safety.ZONE_CAUTION, "CAUTION"),
            (7.0, safety.ZONE_WARNING, "WARNING"),
            (3.0, safety.ZONE_STOP, "CRITICAL"),
        ),
    )
    def test_proximity_zone_mapping(self, distance: float, expected_zone: str,
                                    expected_level: str) -> None:
        state = safety.evaluate_proximity(self._sanitise(build_frame(3, proximity_m=distance)))
        assert state.zone == expected_zone
        assert state.level == expected_level
        assert state.nearest_m == pytest.approx(distance, abs=0.01)

    def test_proximity_tracks_worst_case_and_current_reading(self) -> None:
        frame = build_frame(3, proximity_m=[20.0, 3.4, 22.0])
        state = safety.evaluate_proximity(self._sanitise(frame))
        assert state.zone == safety.ZONE_STOP          # window worst case
        assert state.current_zone == safety.ZONE_CLEAR  # latest reading
        assert state.stop_events == 1

    def test_proximity_fails_safe_on_a_dead_sensor(self) -> None:
        frame = self._sanitise(build_frame(4, proximity_m=-1.0))  # impossible -> rejected
        state = safety.evaluate_proximity(frame)
        assert state.zone == safety.ZONE_UNKNOWN
        assert state.level == "UNKNOWN"
        assert "occupied" in state.message.lower() or "fault" in state.message.lower()

    @pytest.mark.parametrize("reading", (float("nan"), None, "abc", float("inf"),
                                         float("-inf"), object()))
    def test_unusable_single_readings_are_unknown(self, reading) -> None:
        zone, level = safety.proximity_zone_from_reading(reading)
        assert zone == safety.ZONE_UNKNOWN and level == "UNKNOWN"

    def test_negative_reading_is_treated_as_an_intrusion(self) -> None:
        """
        Fail-safe: a corrupt negative distance must resolve to STOP, never to CLEAR.

        (Ingestion rejects negative readings outright; this guards the classifier itself, so a
        future feed that bypasses the sanitiser cannot make the radar look clear.)
        """
        zone, level = safety.proximity_zone_from_reading(-1.0)
        assert zone == safety.ZONE_STOP and level == "CRITICAL"

    def test_snapshot_is_fail_safe_for_empty_telemetry(self, null_frame) -> None:
        snapshot = safety.safety_snapshot(null_frame)
        assert snapshot.overall_level == "UNKNOWN"
        assert snapshot.all_clear is False
        assert snapshot.blockers and snapshot.safety_score == 0
        assert "unknown" in snapshot.headline().lower()

    def test_snapshot_folds_indicators_conservatively(self) -> None:
        frame = build_frame(6, seatbelt_latched=True, proximity_m=[20.0] * 4 + [4.0, 20.0])
        snapshot = safety.safety_snapshot(self._sanitise(frame))
        assert snapshot.overall_level == "CRITICAL"
        assert snapshot.all_clear is False
        assert snapshot.blockers
        assert snapshot.safety_score < 100

    def test_safety_score_monotonicity(self) -> None:
        best = safety.safety_score(safety.SeatbeltState(level="OK"), safety.ProximityState(level="OK"), 1.0)
        caution = safety.safety_score(safety.SeatbeltState(level="CAUTION"), safety.ProximityState(level="CAUTION"), 1.0)
        worst = safety.safety_score(safety.SeatbeltState(level="CRITICAL"), safety.ProximityState(level="CRITICAL"), 0.0)
        unknown = safety.safety_score(safety.SeatbeltState(level="UNKNOWN"), safety.ProximityState(level="UNKNOWN"), 0.5)
        assert best == 100
        assert best > caution > worst == 0
        assert 0 <= unknown <= best
        assert all(0 <= score <= 100 for score in (best, caution, worst, unknown))

    def test_snapshot_on_the_simulated_shift(self, analysed: pd.DataFrame) -> None:
        snapshot = safety.safety_snapshot(analysed)
        assert snapshot.samples > 0
        assert snapshot.window_start is not None and snapshot.window_end is not None
        assert snapshot.seatbelt.unlatched_moving_seconds > 0  # scripted fault in the demo data

    def test_snapshot_headline_is_plain_text(self, analysed: pd.DataFrame) -> None:
        headline = safety.safety_snapshot(analysed).headline()
        assert "<" not in headline and ">" not in headline


# ======================================================================================
# Formatting helpers
# ======================================================================================
class TestFormatting:
    """Labels the operator reads must be correct, plain text, and never raise."""

    @pytest.mark.parametrize(
        "hours,expected",
        ((0.0, "0 min"), (0.5, "30 min"), (1.0, "1 h"), (2.25, "2 h 15 min"), (-3.0, "0 min")),
    )
    def test_fmt_hours(self, hours: float, expected: str) -> None:
        assert fmt.fmt_hours(hours) == expected

    @pytest.mark.parametrize(
        "seconds,expected",
        ((45, "45 s"), (90, "1 min 30 s"), (3_600, "1 h"), (7_500, "2 h 05 min")),
    )
    def test_fmt_seconds(self, seconds: float, expected: str) -> None:
        assert fmt.fmt_seconds(seconds) == expected

    def test_number_formatters_handle_bad_input(self) -> None:
        for bad in (None, float("nan"), "not a number", {}):
            assert fmt.fmt_tons(bad) == "-"
            assert fmt.fmt_pct(bad) == "-"
        assert fmt.fmt_tons(1_830) == "1,830 t"
        assert fmt.fmt_tph(182.4) == "182.4 t/h"
        assert fmt.fmt_litres(19.76) == "19.8 L"

    def test_clock_and_delta_formatters(self) -> None:
        assert fmt.fmt_clock(datetime(2026, 9, 23, 14, 5)) == "14:05"
        assert fmt.fmt_clock(pd.Timestamp("2026-09-23 06:00:00")) == "06:00"
        assert fmt.fmt_clock(None) == "-"
        assert fmt.fmt_day(datetime(2026, 9, 23, 14, 5)) == "2026-09-23 14:05"
        assert fmt.fmt_delta(-184.2, "t") == "-184 t"
        assert fmt.fmt_delta(120.0, "t") == "+120 t"

    def test_progress_ratio_is_clamped(self) -> None:
        assert fmt.progress_ratio(50, 100) == pytest.approx(0.5)
        assert fmt.progress_ratio(150, 100) == 1.0
        assert fmt.progress_ratio(-5, 100) == 0.0
        assert fmt.progress_ratio(10, 0) == 0.0        # no ZeroDivisionError
        assert fmt.progress_ratio(None, None) == 0.0

    def test_badges_and_status_lines_are_plain_text(self) -> None:
        for level in ("OK", "INFO", "CAUTION", "WARNING", "CRITICAL", "UNKNOWN", "NO_DATA"):
            badge = fmt.badge(level)
            assert "<" not in badge and ">" not in badge
        assert "WARNING" in fmt.badge("WARNING")
        line = fmt.status_line("Seatbelt", "latched", "OK", note="full window")
        assert "<" not in line
        assert fmt.checklist_row(True, "seatbelt") .startswith("☑")
        assert fmt.checklist_row(False, "seatbelt").startswith("☐")

    def test_truncate_is_bounded(self) -> None:
        assert len(fmt.truncate("x" * 500, 40)) == 40
        assert fmt.truncate("short") == "short"
        assert fmt.truncate(None) == ""

    def test_severity_colours_are_hex(self) -> None:
        for level in ("OK", "CAUTION", "WARNING", "CRITICAL", "UNKNOWN"):
            assert fmt.level_color(level).startswith("#")
        assert fmt.level_color("nonsense") == fmt.LEVEL_COLORS["UNKNOWN"]
