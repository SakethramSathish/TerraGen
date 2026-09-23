"""
tests/test_telemetry.py
=======================
Unit tests for the telemetry pipeline: hardened ingestion, the anomaly engine (including
**boundary-value** tests with physically impossible sensor data), idle/cycle detection and
the KPI maths.

Boundary-value philosophy
-------------------------
A reading outside the physically possible band (``rpm = 9999``, ``proximity_m = -1.0``,
``fuel_level_pct = 250``, ``coolant_temp_c = 900``) is a **broken sensor**, not an event.
The pipeline must therefore:

1. never raise,
2. null the impossible value,
3. raise ``flag_sensor_fault`` on that row, and
4. keep every downstream consumer (dashboard, safety, KPI) working with finite numbers.
"""

from __future__ import annotations

import math

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

import config
from core import safety, telemetry
from tests.conftest import build_frame


# ======================================================================================
# Ingestion & sanitisation
# ======================================================================================
class TestIngestion:
    """Column allow-list, strict casting, de-duplication and the audit digest."""

    def test_unknown_columns_are_dropped(self, frame_factory) -> None:
        frame = frame_factory(3)
        frame["operator_notes"] = "sabotage"
        frame["__proto__"] = 1
        result = telemetry.sanitize_telemetry_frame(frame)
        assert set(result.dropped_columns) == {"operator_notes", "__proto__"}
        assert "operator_notes" not in result.frame.columns

    def test_types_are_strictly_cast(self, frame_factory) -> None:
        frame = frame_factory(3)
        frame["rpm"] = ["1800", "1750.6", "not-a-number"]
        frame["fuel_rate_lph"] = ["31.5", "x", "29"]
        frame["seatbelt_latched"] = ["1", "latched", "maybe"]
        result = telemetry.sanitize_telemetry_frame(frame)
        assert str(result.frame["rpm"].dtype) == "Int64"
        assert str(result.frame["fuel_rate_lph"].dtype) == "float64"
        assert str(result.frame["seatbelt_latched"].dtype) == "boolean"
        assert result.frame["rpm"].iloc[0] == 1800
        assert result.frame["rpm"].iloc[2] is pd.NA or pd.isna(result.frame["rpm"].iloc[2])
        assert bool(result.frame["seatbelt_latched"].iloc[1]) is True
        assert pd.isna(result.frame["seatbelt_latched"].iloc[2])

    @pytest.mark.parametrize(
        "column,value",
        (
            ("rpm", 9_999),
            ("rpm", -5),
            ("proximity_m", -1.0),
            ("fuel_level_pct", 250.0),
            ("coolant_temp_c", 900.0),
            ("hydraulic_psi", 99_999.0),
            ("ground_speed_kph", 400.0),
        ),
    )
    def test_impossible_values_are_nulled_and_flagged(self, column: str, value: float) -> None:
        frame = build_frame(2, **{column: value})
        result = telemetry.sanitize_telemetry_frame(frame)
        assert result.rows_kept == 2
        assert result.frame[column].isna().all(), f"{column}={value} should have been rejected"
        assert bool(result.frame["sensor_fault"].iloc[0]) is True
        assert result.sensor_faults.get(column, 0) == 2

    def test_missing_required_columns_are_reported_not_fatal(self, frame_factory) -> None:
        frame = frame_factory(2).drop(columns=["hydraulic_psi", "proximity_m"])
        result = telemetry.sanitize_telemetry_frame(frame)
        assert set(result.missing_required) == {"hydraulic_psi", "proximity_m"}
        assert result.rows_kept == 2  # rows survive, flagged as sensor faults
        assert result.frame["sensor_fault"].all()

    def test_duplicate_samples_are_removed(self, frame_factory) -> None:
        frame = frame_factory(4)
        doubled = pd.concat([frame, frame], ignore_index=True)
        result = telemetry.sanitize_telemetry_frame(doubled)
        assert result.rows_kept == 4
        assert any("duplicate" in issue for issue in result.issues)

    def test_rows_are_ordered_by_timestamp(self, frame_factory) -> None:
        frame = frame_factory(4).iloc[::-1].reset_index(drop=True)
        result = telemetry.sanitize_telemetry_frame(frame)
        assert result.frame["timestamp"].is_monotonic_increasing

    def test_unparsable_timestamps_get_a_synthetic_grid(self, frame_factory) -> None:
        frame = frame_factory(3)
        frame["timestamp"] = ["garbage", "worse", ""]
        result = telemetry.sanitize_telemetry_frame(frame)
        assert result.frame["timestamp"].notna().all()
        assert any("synthetic" in issue for issue in result.issues)

    def test_digest_is_stable_and_content_sensitive(self, frame_factory) -> None:
        frame = frame_factory(4)
        first = telemetry.sanitize_telemetry_frame(frame)
        second = telemetry.sanitize_telemetry_frame(frame.copy())
        assert first.digest == second.digest
        changed = frame.copy()
        changed.loc[0, "rpm"] = 1_801
        assert telemetry.sanitize_telemetry_frame(changed).digest != first.digest

    def test_row_names_are_not_trusted(self, frame_factory) -> None:
        """Attacker-controlled headers must not create columns or crash the ingest."""
        frame = frame_factory(2)
        frame.columns = ["<img src=x onerror=alert(1)>" if i == 2 else c
                         for i, c in enumerate(frame.columns)]
        result = telemetry.sanitize_telemetry_frame(frame)
        assert result.rows_kept == 2
        for column in result.frame.columns:
            assert "<" not in str(column)

    @pytest.mark.parametrize("bad", [None, pd.DataFrame(), "not a frame", 42, [1, 2, 3]])
    def test_non_frame_inputs_return_an_empty_result(self, bad) -> None:
        result = telemetry.sanitize_telemetry_frame(bad)  # type: ignore[arg-type]
        assert result.rows_kept == 0
        assert result.frame.empty
        assert result.issues

    def test_missing_file_is_reported(self, tmp_path) -> None:
        result = telemetry.load_telemetry_csv(tmp_path / "nope.csv")
        assert result.rows_kept == 0
        assert any("not found" in issue for issue in result.issues)

    def test_malformed_csv_does_not_raise(self) -> None:
        result = telemetry.load_telemetry_csv(b"\x00\x01\x02 not,csv,at,all\n\n\xff")
        assert isinstance(result, telemetry.TelemetryIngestResult)

    def test_empty_frame_has_the_full_schema(self) -> None:
        frame = telemetry.empty_telemetry_frame()
        assert frame.empty
        for spec in config.TELEMETRY_SCHEMA:
            assert spec.name in frame.columns
        assert "sensor_fault" in frame.columns


# ======================================================================================
# Anomaly engine
# ======================================================================================
class TestAnomalyEngine:
    """One test per rule, then the composite/severity behaviour."""

    def _analyse(self, frame: pd.DataFrame) -> pd.DataFrame:
        return telemetry.detect_anomalies(telemetry.sanitize_telemetry_frame(frame).frame)

    def test_healthy_window_raises_nothing(self) -> None:
        analysed = self._analyse(build_frame(10))
        assert bool(analysed["is_anomaly"].any()) is False
        assert set(analysed["severity"]) == {"OK"}

    def test_over_rev_detected(self) -> None:
        analysed = self._analyse(build_frame(3, rpm=[1_800, 2_400, 1_800]))
        assert bool(analysed["flag_over_rev"].iloc[1]) is True
        assert analysed["severity"].iloc[1] == "WARNING"
        assert "over-speed" in analysed["reasons"].iloc[1]

    def test_low_and_critical_fuel(self) -> None:
        analysed = self._analyse(build_frame(3, fuel_level_pct=[50.0, 12.0, 5.0]))
        assert bool(analysed["flag_low_fuel"].iloc[1]) is True
        assert bool(analysed["flag_critical_fuel"].iloc[2]) is True
        assert analysed["severity"].iloc[2] == "WARNING"

    def test_coolant_over_temperature_is_critical(self) -> None:
        analysed = self._analyse(build_frame(2, coolant_temp_c=[88.0, 106.0]))
        assert bool(analysed["flag_coolant_overtemp"].iloc[1]) is True
        assert analysed["severity"].iloc[1] == "CRITICAL"

    def test_low_oil_pressure_only_counts_while_running(self) -> None:
        running = self._analyse(build_frame(2, oil_pressure_kpa=[400.0, 90.0], rpm=[1_800, 1_800]))
        assert bool(running["flag_low_oil_pressure"].iloc[1]) is True
        stopped = self._analyse(build_frame(2, oil_pressure_kpa=[400.0, 90.0], rpm=[0, 0]))
        assert bool(stopped["flag_low_oil_pressure"].iloc[1]) is False

    def test_hydraulic_overpressure(self) -> None:
        analysed = self._analyse(build_frame(2, hydraulic_psi=[4_200.0, 5_900.0]))
        assert bool(analysed["flag_hydraulic_overpressure"].iloc[1]) is True

    def test_hydraulic_underpressure_requires_demand(self) -> None:
        loaded = self._analyse(build_frame(2, hydraulic_psi=[4_200.0, 900.0], payload_tons=[6.0, 6.0]))
        assert bool(loaded["flag_hydraulic_underpressure"].iloc[1]) is True
        empty = self._analyse(build_frame(2, hydraulic_psi=[4_200.0, 900.0], payload_tons=[0.0, 0.0],
                                          ground_speed_kph=[0.0, 0.0]))
        assert bool(empty["flag_hydraulic_underpressure"].iloc[1]) is False

    def test_seatbelt_unlatched_while_moving_is_critical(self) -> None:
        analysed = self._analyse(
            build_frame(3, seatbelt_latched=[True, False, False],
                        ground_speed_kph=[3.0, 8.0, 0.0])
        )
        assert bool(analysed["flag_seatbelt_while_moving"].iloc[1]) is True
        assert analysed["severity"].iloc[1] == "CRITICAL"
        # released while parked is not flagged by this rule
        assert bool(analysed["flag_seatbelt_while_moving"].iloc[2]) is False

    @pytest.mark.parametrize(
        "distance,expected_intrusion,expected_critical",
        ((20.0, False, False), (12.0, False, False), (7.5, True, False), (3.0, True, True)),
    )
    def test_proximity_zones(self, distance: float, expected_intrusion: bool,
                             expected_critical: bool) -> None:
        analysed = self._analyse(build_frame(1, proximity_m=distance))
        assert bool(analysed["flag_proximity_intrusion"].iloc[0]) is expected_intrusion
        assert bool(analysed["flag_proximity_critical"].iloc[0]) is expected_critical

    def test_injected_idle_window_is_detected(self, analysed: pd.DataFrame) -> None:
        windows = telemetry.detect_idle_windows(analysed)
        assert windows, "the simulated shift contains a scripted 8-minute idle"
        longest = max(windows, key=lambda window: window.duration_s)
        assert longest.duration_s >= config.THRESHOLDS.high_idle_seconds
        assert longest.avg_rpm <= config.THRESHOLDS.idle_rpm_max
        assert longest.fuel_burned_l > 0

    def test_short_idle_is_not_an_event(self) -> None:
        """A 2-minute truck wait is normal operation, not fuel waste."""
        frame = build_frame(24, rpm=900, ground_speed_kph=0.0, payload_tons=0.0,
                            hydraulic_psi=450.0, fuel_rate_lph=6.0)
        analysed = self._analyse(frame)
        assert telemetry.detect_idle_windows(analysed) == []
        assert bool(analysed["is_anomaly"].any()) is False

    def test_sustained_idle_is_flagged_on_rows(self) -> None:
        frame = build_frame(80, rpm=880, ground_speed_kph=0.0, payload_tons=0.0,
                            hydraulic_psi=430.0, fuel_rate_lph=6.2)
        analysed = self._analyse(frame)
        assert bool(analysed["flag_high_idle"].all()) is True
        assert analysed["severity"].iloc[0] == "INFO"

    def test_severity_folds_to_the_worst_active_flag(self) -> None:
        analysed = self._analyse(
            build_frame(2, coolant_temp_c=[88.0, 120.0], fuel_level_pct=[70.0, 5.0], proximity_m=[20.0, 2.0])
        )
        assert analysed["severity"].iloc[1] == "CRITICAL"
        assert analysed["anomaly_count"].iloc[1] >= 3
        assert ";" in analysed["reasons"].iloc[1]

    def test_anomaly_log_is_static_text_only(self, analysed: pd.DataFrame) -> None:
        log = telemetry.anomaly_log(analysed, limit=10)
        assert not log.empty
        assert "<" not in log.to_csv(index=False)  # no markup can reach the log view


# ======================================================================================
# Boundary values / hostile feeds
# ======================================================================================
class TestBoundaryValues:
    """Extreme synthetic data must be handled safely by every entry point."""

    #: ``(case, expect_sensor_fault)`` - ``expect_fault`` separates "physically impossible"
    #: (a broken sensor: 9 999 rpm) from "legitimately extreme" (engine stopped: 0 rpm).
    EXTREME_CASES: tuple[tuple[dict[str, float], bool], ...] = (
        ({"rpm": 9_999}, True),
        ({"rpm": 0}, False),
        ({"rpm": -1}, True),
        ({"proximity_m": -1.0}, True),
        ({"proximity_m": 0.0}, False),
        ({"proximity_m": 1e9}, True),
        ({"fuel_level_pct": 250.0}, True),
        ({"fuel_level_pct": -20.0}, True),
        ({"hydraulic_psi": 0.0}, False),
        ({"hydraulic_psi": 1e6}, True),
        ({"coolant_temp_c": 900.0}, True),
        ({"coolant_temp_c": -273.0}, True),
        ({"ground_speed_kph": 400.0}, True),
        ({"payload_tons": 500.0}, True),
        ({"payload_tons": -5.0}, True),
        # Optional channels are nulled and counted in ``sensor_faults`` but do not raise the
        # row-level fault flag (see TestBoundaryValues.test_optional_channels_are_nulled_only).
        ({"oil_pressure_kpa": -10.0}, False),
        ({"hydraulic_oil_temp_c": 500.0}, False),
    )

    @pytest.mark.parametrize(
        "case,expect_fault", EXTREME_CASES, ids=lambda value: str(value)[:24]
    )
    def test_extremes_never_crash_and_never_produce_non_finite_output(
        self, case: dict[str, float], expect_fault: bool
    ) -> None:
        frame = build_frame(4, **case)
        result = telemetry.sanitize_telemetry_frame(frame)
        analysed = telemetry.detect_anomalies(result.frame)
        kpis = telemetry.compute_kpis(analysed)
        snapshot = safety.safety_snapshot(analysed)
        status = telemetry.machine_status(analysed)

        assert result.rows_kept == 4
        assert bool(result.frame["sensor_fault"].any()) is expect_fault
        for column in ("rows", "tons_hauled", "cycles", "avg_cycle_s", "tons_per_hour",
                       "fuel_burned_l", "fuel_per_ton_l", "idle_pct", "avg_rpm", "max_rpm",
                       "p95_hydraulic_psi", "anomaly_rows", "anomaly_events", "critical_events",
                       "window_hours"):
            assert math.isfinite(float(getattr(kpis, column))), column
        assert isinstance(status, str) and status
        assert snapshot.overall_level in {"OK", "CAUTION", "WARNING", "CRITICAL", "UNKNOWN"}

    def test_optional_channels_are_nulled_only(self) -> None:
        """An out-of-range *optional* sensor is dropped without condemning the whole row."""
        frame = build_frame(2, oil_pressure_kpa=-10.0, hydraulic_oil_temp_c=500.0)
        result = telemetry.sanitize_telemetry_frame(frame)
        assert result.frame["oil_pressure_kpa"].isna().all()
        assert result.frame["hydraulic_oil_temp_c"].isna().all()
        assert result.sensor_faults["oil_pressure_kpa"] == 2
        assert bool(result.frame["sensor_fault"].any()) is False

    def test_all_nan_frame(self, frame_factory) -> None:
        frame = frame_factory(3).drop(columns=["rpm", "proximity_m", "payload_tons"])
        frame["rpm"] = np.nan
        frame["proximity_m"] = np.nan
        frame["payload_tons"] = np.nan
        analysed = telemetry.detect_anomalies(telemetry.sanitize_telemetry_frame(frame).frame)
        assert telemetry.compute_kpis(analysed).rows == 3
        assert safety.safety_snapshot(analysed).overall_level == "UNKNOWN"
        assert telemetry.machine_status(analysed).startswith(("FAULT", "NO_DATA", "STOPPED"))
        assert telemetry.detect_idle_windows(analysed) == []
        assert telemetry.detect_haul_cycles(analysed) == []

    def test_mixed_valid_and_invalid_rows(self) -> None:
        frame = build_frame(3)
        frame.loc[1, "rpm"] = 9_999
        frame.loc[1, "proximity_m"] = -1.0
        analysed = telemetry.detect_anomalies(telemetry.sanitize_telemetry_frame(frame).frame)
        assert bool(analysed["sensor_fault"].iloc[1]) is True
        assert bool(analysed["sensor_fault"].iloc[0]) is False
        assert bool(analysed["flag_proximity_critical"].iloc[1]) is False  # unknown != intrusion

    def test_zero_and_negative_shift_targets_are_guarded(self) -> None:
        analysed = telemetry.detect_anomalies(telemetry.sanitize_telemetry_frame(build_frame(4)).frame)
        kpis = telemetry.compute_kpis(analysed)
        from core import dashboard

        for target in (0.0, -100.0):
            plan = dashboard.build_shift_plan(target_tons=target)
            state = dashboard.evaluate_shift(plan, kpis)
            assert state.progress_pct >= 0.0
            assert math.isfinite(state.progress_pct)

    def test_clock_skew_does_not_produce_negative_infrastructure(self) -> None:
        analysed = telemetry.detect_anomalies(
            telemetry.sanitize_telemetry_frame(build_frame(3)).frame
        )
        from core import dashboard

        plan = dashboard.build_shift_plan()
        state = dashboard.evaluate_shift(plan, telemetry.compute_kpis(analysed),
                                         now=plan.start - timedelta(hours=5))
        assert state.elapsed_hours == 0.0
        assert state.remaining_hours >= 0.0

    def test_naive_and_aware_style_inputs(self) -> None:
        frame = build_frame(2, start="2026-09-23T06:00:00+05:30")
        result = telemetry.sanitize_telemetry_frame(frame)
        assert result.rows_kept == 2
        assert telemetry.compute_kpis(result.frame).rows == 2


# ======================================================================================
# KPIs, cycles and status
# ======================================================================================
class TestKpisAndCycles:
    """Production maths: cycles, tonnage, idle share, machine status."""

    def test_cycles_and_tonnage_from_a_known_pattern(self) -> None:
        # Three loads of 8 t, each spanning two samples.
        payload = [0.0, 8.0, 8.0, 0.0, 0.0, 8.0, 8.0, 0.0, 0.0, 8.0, 8.0, 0.0]
        frame = build_frame(12, payload_tons=payload, rpm=[900 if p < 1 else 1_800 for p in payload])
        kpis = telemetry.compute_kpis(telemetry.sanitize_telemetry_frame(frame).frame)
        assert kpis.cycles == 3
        assert kpis.tons_hauled == pytest.approx(24.0, abs=0.01)

    def test_tons_per_hour_uses_the_window_length(self) -> None:
        analysed = telemetry.sanitize_telemetry_frame(
            telemetry.simulate_telemetry(telemetry.SimulationConfig(minutes=60))
        ).frame
        kpis = telemetry.compute_kpis(analysed)
        assert kpis.window_hours == pytest.approx(1.0, abs=0.01)
        assert kpis.tons_per_hour == pytest.approx(kpis.tons_hauled, rel=0.05)
        assert kpis.tons_hauled > 100  # a 320 moving real tonnage in an hour

    def test_fuel_integration_matches_the_rate(self) -> None:
        # 721 samples at a 5 s interval span exactly one hour at a constant 36 L/h.
        frame = build_frame(721, fuel_rate_lph=36.0, rpm=1_800, payload_tons=6.0)
        kpis = telemetry.compute_kpis(telemetry.sanitize_telemetry_frame(frame).frame)
        # 721 samples at 5 s = 3600 s = 1 h at 36 L/h
        assert kpis.fuel_burned_l == pytest.approx(36.0, rel=0.02)

    @pytest.mark.parametrize(
        "kwargs,expected_prefix",
        (
            ({}, "RUNNING"),
            ({"rpm": 0, "ground_speed_kph": 0.0, "payload_tons": 0.0}, "STOPPED"),
            ({"rpm": 850, "ground_speed_kph": 0.0, "payload_tons": 0.0}, "IDLE"),
            ({"rpm": 1_800, "ground_speed_kph": 7.0, "payload_tons": 0.0}, "RUNNING"),
        ),
    )
    def test_machine_status(self, kwargs, expected_prefix) -> None:
        analysed = telemetry.detect_anomalies(telemetry.sanitize_telemetry_frame(build_frame(3, **kwargs)).frame)
        assert telemetry.machine_status(analysed).startswith(expected_prefix)

    def test_machine_status_on_empty_frame(self, null_frame) -> None:
        assert telemetry.machine_status(null_frame).startswith("NO_DATA")

    def test_count_events_counts_runs_not_samples(self) -> None:
        mask = [False, True, True, False, True, False, False]
        assert telemetry.count_events(mask) == 2
        assert telemetry.count_events([]) == 0
        assert telemetry.count_events([True, True, True]) == 1

    def test_kpis_on_empty_frame(self, null_frame) -> None:
        kpis = telemetry.compute_kpis(null_frame)
        assert kpis.rows == 0 and kpis.status == "NO_DATA"
        assert telemetry.summarize_flags(null_frame).empty
        assert telemetry.anomaly_log(null_frame).empty

    def test_simulator_is_deterministic_and_bounded(self) -> None:
        first = telemetry.simulate_telemetry(telemetry.SimulationConfig(minutes=20, seed=11))
        second = telemetry.simulate_telemetry(telemetry.SimulationConfig(minutes=20, seed=11))
        third = telemetry.simulate_telemetry(telemetry.SimulationConfig(minutes=20, seed=12))
        pd.testing.assert_frame_equal(first, second)
        assert not first["rpm"].equals(third["rpm"])
        assert first["rpm"].between(0, 4_000).all()
        assert first["proximity_m"].between(0, 30).all()
        assert first["payload_tons"].between(0, 12.5).all()

    def test_simulated_faults_are_present_for_the_demo(self, analysed: pd.DataFrame) -> None:
        summary = telemetry.summarize_flags(analysed)
        flags = set(summary["flag"])
        assert {"flag_high_idle", "flag_over_rev", "flag_proximity_critical",
                "flag_seatbelt_while_moving", "flag_critical_fuel"} <= flags

    def test_sensor_fault_injection_creates_a_fault_row(self) -> None:
        """The demo 'inject sensor fault' toggle must produce a real, flagged fault."""
        frame = telemetry.simulate_telemetry(
            telemetry.SimulationConfig(minutes=20, inject_sensor_fault=True)
        )
        result = telemetry.sanitize_telemetry_frame(frame)
        assert result.sensor_faults, "no sensor fault was recorded"
        assert bool(result.frame["sensor_fault"].any()) is True
        assert telemetry.machine_status(telemetry.detect_anomalies(result.frame))

    def test_bundled_sample_csv_ingests_cleanly(self) -> None:
        if not config.SAMPLE_CSV.exists():  # pragma: no cover - CI without generated sample
            pytest.skip("sample CSV not generated (run scripts/generate_sample_data.py)")
        result = telemetry.load_telemetry_csv(config.SAMPLE_CSV)
        assert result.rows_kept > 1_000
        assert result.frame["sensor_fault"].sum() == 0
        assert result.digest
