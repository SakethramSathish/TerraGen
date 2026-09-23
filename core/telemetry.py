"""
core/telemetry.py
=================
Simulated/real **telemetry ingestion pipeline** + deterministic **anomaly engine** for the
CAT Smart Operator Assistant.

Pipeline overview
-----------------
::

    CSV / dict / DataFrame      (untrusted: operator upload or a compromised CAN gateway)
            │
            ▼
    load_telemetry_csv()        size cap, row cap, read everything as *strings*
            │
            ▼
    sanitize_telemetry_frame()  column allow-list ▸ strict dtype casting ▸ hard-bound
            │                   clipping ▸ timestamp parsing ▸ de-duplication ▸ SHA-256
            ▼                   content digest for the audit chain
    detect_anomalies()          operational thresholds ▸ per-row flags ▸ severity fold
            │
            ├─► detect_idle_windows()   sustained high-idle events
            ├─► detect_haul_cycles()    productivity / tonnage accounting
            ├─► compute_kpis()          dashboard metrics
            └─► machine_status()        RUNNING / IDLE / STOPPED / FAULT

Security notes
--------------
* **Strict typing is a security control**: every cell is re-parsed with
  :func:`core.security.coerce_numeric` after being read as text, so a payload such as
  ``"1200; DROP TABLE machines;--"`` simply fails to parse and becomes a sensor fault.
  Nothing is ever ``eval``-ed, interpolated into SQL, or written to disk verbatim.
* Values outside *physically impossible* bounds (e.g. RPM = 9999, proximity = -1.0) are
  nulled and flagged ``SENSOR_FAULT`` instead of being clamped into fake "good" data -
  a corrupt frame must never masquerade as a valid reading.
* The engine is total: it returns a well-formed (possibly empty) frame for *any* input and
  never raises into the UI, which is what keeps the dashboard alive on bad data.
"""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from config import (
    MAX_TELEMETRY_ROWS,
    MAX_UPLOAD_BYTES,
    MACHINE_ID_RE,
    SCHEMA_BY_NAME,
    SEVERITY_ORDER,
    TELEMETRY_SCHEMA,
    THRESHOLDS,
    Thresholds,
)
from core.security import coerce_numeric, flatten_for_log, normalize_unicode, validate_sensor_value

# --------------------------------------------------------------------------------------
# Column / flag vocabulary
# --------------------------------------------------------------------------------------
FLAG_COLUMNS: tuple[str, ...] = (
    "flag_over_rev",
    "flag_high_idle",
    "flag_idle_excess_fuel",
    "flag_low_fuel",
    "flag_critical_fuel",
    "flag_hydraulic_overpressure",
    "flag_hydraulic_underpressure",
    "flag_hydraulic_overtemp",
    "flag_coolant_overtemp",
    "flag_low_oil_pressure",
    "flag_seatbelt_while_moving",
    "flag_proximity_intrusion",
    "flag_proximity_critical",
    "flag_sensor_fault",
)

#: Human labels used in the anomaly log (static strings - never user data).
FLAG_LABELS: dict[str, str] = {
    "flag_over_rev": "Engine over-speed",
    "flag_high_idle": "Sustained high idle",
    "flag_idle_excess_fuel": "Excess fuel burn at idle",
    "flag_low_fuel": "Low fuel level",
    "flag_critical_fuel": "Critical fuel level",
    "flag_hydraulic_overpressure": "Hydraulic over-pressure",
    "flag_hydraulic_underpressure": "Hydraulic under-pressure",
    "flag_hydraulic_overtemp": "Hydraulic oil over-temperature",
    "flag_coolant_overtemp": "Coolant over-temperature",
    "flag_low_oil_pressure": "Low engine oil pressure",
    "flag_seatbelt_while_moving": "Seatbelt unlatched while moving",
    "flag_proximity_intrusion": "Proximity warning zone entered",
    "flag_proximity_critical": "Proximity STOP zone entered",
    "flag_sensor_fault": "Sensor fault / invalid reading",
}

#: Severity assigned to each individual flag.
FLAG_SEVERITY: dict[str, str] = {
    "flag_over_rev": "WARNING",
    "flag_high_idle": "INFO",
    "flag_idle_excess_fuel": "CAUTION",
    "flag_low_fuel": "CAUTION",
    "flag_critical_fuel": "WARNING",
    "flag_hydraulic_overpressure": "WARNING",
    "flag_hydraulic_underpressure": "WARNING",
    "flag_hydraulic_overtemp": "WARNING",
    "flag_coolant_overtemp": "CRITICAL",
    "flag_low_oil_pressure": "CRITICAL",
    "flag_seatbelt_while_moving": "CRITICAL",
    "flag_proximity_intrusion": "WARNING",
    "flag_proximity_critical": "CRITICAL",
    "flag_sensor_fault": "CAUTION",
}

MACHINE_STATUSES: tuple[str, ...] = ("NO_DATA", "RUNNING", "IDLE", "STOPPED", "FAULT")


# --------------------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------------------
@dataclass
class TelemetryIngestResult:
    """Outcome of a full ingest + sanitisation pass."""

    frame: pd.DataFrame
    rows_received: int = 0
    rows_kept: int = 0
    rows_rejected: int = 0
    truncated: bool = False
    dropped_columns: tuple[str, ...] = ()
    missing_required: tuple[str, ...] = ()
    sensor_faults: dict[str, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    digest: str = ""

    @property
    def ok(self) -> bool:
        """True when at least one usable row survived sanitisation."""
        return not self.frame.empty

    def summary(self) -> str:
        """One-line status for the ingest audit panel."""
        return (
            f"{self.rows_kept}/{self.rows_received} rows kept · "
            f"{self.rows_rejected} rejected · "
            f"{sum(self.sensor_faults.values())} field fault(s) · "
            f"digest {self.digest[:12] or 'n/a'}"
        )


@dataclass(frozen=True)
class IdleWindow:
    """A sustained-idle event (candidate fuel-waste / operator-coaching insight)."""

    start: pd.Timestamp
    end: pd.Timestamp
    duration_s: float
    avg_rpm: float
    fuel_burned_l: float

    def describe(self) -> str:
        return (
            f"{self.duration_s / 60.0:.1f} min idle "
            f"(avg {self.avg_rpm:.0f} rpm, {self.fuel_burned_l:.1f} L burned)"
        )


@dataclass(frozen=True)
class HaulCycle:
    """One load-haul-dump cycle detected from payload telemetry."""

    start: pd.Timestamp
    end: pd.Timestamp
    peak_payload_tons: float
    duration_s: float


@dataclass
class KpiSummary:
    """Dashboard KPIs computed from one sanitised window."""

    rows: int = 0
    window_start: pd.Timestamp | None = None
    window_end: pd.Timestamp | None = None
    window_hours: float = 0.0
    tons_hauled: float = 0.0
    cycles: int = 0
    avg_cycle_s: float = 0.0
    tons_per_hour: float = 0.0
    fuel_burned_l: float = 0.0
    fuel_per_ton_l: float = 0.0
    idle_pct: float = 0.0
    avg_rpm: float = 0.0
    max_rpm: float = 0.0
    p95_hydraulic_psi: float = 0.0
    anomaly_rows: int = 0
    anomaly_events: int = 0
    critical_events: int = 0
    status: str = "NO_DATA"


# --------------------------------------------------------------------------------------
# Empty frame helper
# --------------------------------------------------------------------------------------
def empty_telemetry_frame() -> pd.DataFrame:
    """
    A correctly-typed, zero-row telemetry frame.

    Guarantees the UI can always render (charts, metrics, tables) even when ingestion
    failed completely - no ``None``/``KeyError`` paths in the dashboard.
    """
    data: dict[str, pd.Series] = {}
    for spec in TELEMETRY_SCHEMA:
        if spec.dtype == "timestamp":
            data[spec.name] = pd.Series([], dtype="datetime64[ns]")
        elif spec.dtype == "string":
            data[spec.name] = pd.Series([], dtype="object")
        elif spec.dtype == "bool":
            data[spec.name] = pd.Series([], dtype="boolean")
        elif spec.dtype == "int":
            data[spec.name] = pd.Series([], dtype="Int64")
        else:
            data[spec.name] = pd.Series([], dtype="float64")
    frame = pd.DataFrame(data)
    frame["sensor_fault"] = pd.Series([], dtype="bool")
    return frame


# --------------------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------------------
def load_telemetry_csv(
    source: str | Path | bytes | bytearray | io.IOBase,
    *,
    max_rows: int = MAX_TELEMETRY_ROWS,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> TelemetryIngestResult:
    """
    Read an operator-supplied CSV **defensively** and hand it to the sanitiser.

    Controls applied before pandas sees a single value:

    * byte-size cap (``max_bytes``) - refuses oversized uploads outright;
    * row cap (``max_rows``) - extra rows are discarded and reported (DoS protection);
    * everything is read as ``str`` (``dtype=str``) so no pandas type inference can be
      influenced by attacker-controlled content;
    * only whitelisted columns are requested (``usecols``) - unknown columns never load.

    Returns a :class:`TelemetryIngestResult`; CSVs that cannot be parsed at all yield an
    empty frame plus an explanatory issue instead of an exception.
    """
    issues: list[str] = []
    payload: Any

    if isinstance(source, (bytes, bytearray)):
        raw_bytes = bytes(source)
        if len(raw_bytes) > max_bytes:
            return TelemetryIngestResult(
                frame=empty_telemetry_frame(),
                issues=[f"upload rejected: {len(raw_bytes)} bytes exceeds {max_bytes} byte cap"],
            )
        payload = io.BytesIO(raw_bytes)
    elif isinstance(source, io.IOBase):
        payload = source
    else:
        path = Path(source)
        if not path.exists():
            return TelemetryIngestResult(
                frame=empty_telemetry_frame(), issues=[f"file not found: {path.name}"]
            )
        size = path.stat().st_size
        if size > max_bytes:
            return TelemetryIngestResult(
                frame=empty_telemetry_frame(),
                issues=[f"file rejected: {size} bytes exceeds {max_bytes} byte cap"],
            )
        payload = path

    allowed = [spec.name for spec in TELEMETRY_SCHEMA]
    try:
        raw = pd.read_csv(
            payload,
            dtype=str,              # never let pandas infer attacker-influenced types
            usecols=lambda col: normalize_unicode(str(col)).strip() in allowed,
            nrows=max_rows + 1,     # +1 so truncation can be detected and reported
            skip_blank_lines=True,
            keep_default_na=False,
            encoding="utf-8",
            encoding_errors="replace",
            on_bad_lines="skip",    # malformed row never aborts the whole shift file
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        return TelemetryIngestResult(
            frame=empty_telemetry_frame(),
            issues=[f"CSV parse failed safely: {type(exc).__name__}"],
        )

    raw.columns = [normalize_unicode(str(c)).strip() for c in raw.columns]
    truncated = len(raw) > max_rows
    if truncated:
        raw = raw.iloc[:max_rows]
        issues.append(f"row cap hit: only the first {max_rows} rows were ingested")

    result = sanitize_telemetry_frame(raw, sample_interval_s=THRESHOLDS.sample_interval_s)
    result.truncated = result.truncated or truncated
    result.issues = issues + result.issues
    return result


def sanitize_telemetry_frame(
    df: pd.DataFrame | None,
    *,
    sample_interval_s: float = THRESHOLDS.sample_interval_s,
    max_rows: int = MAX_TELEMETRY_ROWS,
) -> TelemetryIngestResult:
    """
    Validate + strictly cast a telemetry DataFrame.

    Steps
    -----
    1. **Shape guard** - ``None``/non-DataFrame/empty input returns an empty typed frame.
    2. **Column allow-list** - unknown columns are dropped (reported, never trusted).
    3. **Row cap** - at most ``max_rows`` rows survive.
    4. **Strict casting** - ``rpm`` -> nullable ``Int64``, ``seatbelt_latched`` -> nullable
       ``boolean``, the rest -> ``float64``; unparsable cells become ``NA``.
    5. **Hard-bound check** - readings outside physically possible ranges are nulled and
       counted as sensor faults (RPM 9999, proximity -1.0, fuel 250 %...).
    6. **Timestamp repair** - unparsable timestamps are rebuilt from a synthetic grid so
       downstream time-series logic still works, and the substitution is reported.
    7. **De-duplication + ordering** - last write wins on ``(timestamp, machine_id)``.
    8. **Digest** - SHA-256 over the canonical frame, for the tamper-evident audit chain.

    The function is total: it always returns a :class:`TelemetryIngestResult`.
    """
    result = TelemetryIngestResult(frame=empty_telemetry_frame())

    if df is None or not isinstance(df, pd.DataFrame):
        result.issues.append("no telemetry supplied (expected a DataFrame)")
        return result
    if df.empty:
        result.issues.append("telemetry frame is empty")
        return result

    working = df.copy(deep=True)
    working.columns = [normalize_unicode(str(c)).strip() for c in working.columns]
    result.rows_received = int(len(working))

    if len(working) > max_rows:
        working = working.iloc[:max_rows]
        result.truncated = True
        result.issues.append(f"row cap hit: truncated to {max_rows} rows")

    # --- 2. column allow-list ------------------------------------------------------
    known = set(SCHEMA_BY_NAME)
    unknown = [c for c in working.columns if c not in known]
    if unknown:
        result.dropped_columns = tuple(str(c) for c in unknown)
        working = working.drop(columns=unknown)
        result.issues.append(f"dropped {len(unknown)} non-schema column(s)")

    missing = [c for c in (s.name for s in TELEMETRY_SCHEMA if s.required) if c not in working.columns]
    if missing:
        result.missing_required = tuple(missing)
        result.issues.append("missing required column(s): " + ", ".join(missing))

    out = pd.DataFrame(index=working.index)
    faults: dict[str, int] = {}

    # --- 4/5. strict casting + hard bounds -----------------------------------------
    for spec in TELEMETRY_SCHEMA:
        if spec.name not in working.columns:
            if spec.dtype == "timestamp":
                out[spec.name] = pd.NaT
            elif spec.dtype == "string":
                out[spec.name] = "UNKNOWN"
            elif spec.dtype == "bool":
                out[spec.name] = pd.NA
            elif spec.dtype == "int":
                out[spec.name] = pd.NA
            else:
                out[spec.name] = np.nan
            continue

        column = working[spec.name]

        if spec.dtype == "timestamp":
            parsed = pd.to_datetime(column, errors="coerce", format="mixed")
            bad = int(parsed.isna().sum())
            if bad:
                faults[spec.name] = bad
                result.issues.append(f"{bad} unparsable timestamp(s)")
            out[spec.name] = parsed

        elif spec.dtype == "string":
            cleaned = (
                column.map(lambda v: normalize_unicode(str(v)).strip()[:64])
                .map(lambda v: v if MACHINE_ID_RE.match(v) else "UNKNOWN")
            )
            rejected = int((cleaned == "UNKNOWN").sum())
            if rejected:
                faults[spec.name] = rejected
                result.issues.append(f"{rejected} value(s) failed machine_id validation")
            out[spec.name] = cleaned.astype("object")

        elif spec.dtype == "bool":
            out[spec.name] = _cast_boolean(column, faults, spec.name, result)

        else:
            numeric = column.map(
                lambda v: coerce_numeric(v, spec.hard_min, spec.hard_max)
            )
            parsed = pd.to_numeric(numeric, errors="coerce")
            bad = int(parsed.isna().sum())
            if bad:
                faults[spec.name] = bad
            if spec.dtype == "int":
                out[spec.name] = parsed.round().astype("Int64")
            else:
                out[spec.name] = parsed.astype("float64")

    # --- 6. timestamp repair -------------------------------------------------------
    if out["timestamp"].isna().all():
        base = pd.Timestamp(datetime(2026, 1, 1, 6, 0, 0))
        out["timestamp"] = [base + timedelta(seconds=sample_interval_s * i) for i in range(len(out))]
        result.issues.append("no usable timestamps: synthetic sample grid applied")
    elif out["timestamp"].isna().any():
        out["timestamp"] = out["timestamp"].ffill().bfill()

    # --- 7. de-duplicate + order ---------------------------------------------------
    before = len(out)
    out = out.drop_duplicates(subset=["timestamp", "machine_id"], keep="last")
    dupes = before - len(out)
    if dupes:
        result.issues.append(f"removed {dupes} duplicate sample(s)")

    out = out.sort_values("timestamp", kind="stable").reset_index(drop=True)

    # --- row-level sensor fault composite ------------------------------------------
    required_numeric = [s.name for s in TELEMETRY_SCHEMA if s.required and s.dtype not in {"string", "timestamp"}]
    fault_mask = pd.Series(False, index=out.index)
    for name in required_numeric:
        fault_mask = fault_mask | out[name].isna()
    out["sensor_fault"] = fault_mask.astype(bool)

    rejected_rows = int(out["sensor_fault"].sum())
    out = out  # rows are kept (flagged) rather than dropped: gaps are operationally meaningful

    # --- 8. audit digest -----------------------------------------------------------
    digest = _frame_digest(out)

    result.frame = out
    result.rows_kept = int(len(out))
    result.rows_rejected = rejected_rows
    result.sensor_faults = faults
    result.digest = digest
    return result


def _cast_boolean(
    column: pd.Series,
    faults: dict[str, int],
    column_name: str,
    result: TelemetryIngestResult,
) -> pd.Series:
    """Cast a messy seatbelt-ish column to a nullable boolean with an explicit lexicon."""
    true_tokens = {"1", "true", "yes", "y", "on", "latched", "closed", "buckled"}
    false_tokens = {"0", "false", "no", "n", "off", "unlatched", "open", "unbuckled"}

    def to_bool(value: object) -> Any:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            ok, num = validate_sensor_value(value)
            if ok and num is not None and num in (0.0, 1.0):
                return bool(int(num))
            return pd.NA
        token = normalize_unicode(str(value)).strip().lower()
        if token in true_tokens:
            return True
        if token in false_tokens:
            return False
        return pd.NA

    cast = column.map(to_bool)
    bad = int(pd.isna(cast).sum())
    if bad:
        faults[column_name] = bad
        result.issues.append(f"{bad} value(s) failed seatbelt boolean cast")
    return cast.astype("boolean")


def _frame_digest(frame: pd.DataFrame) -> str:
    """SHA-256 digest of the canonical frame - supports tamper-evident shift records."""
    try:
        hashed = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="uint64")
        return hashlib.sha256(hashed.tobytes()).hexdigest()
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return ""


# --------------------------------------------------------------------------------------
# Working view helpers
# --------------------------------------------------------------------------------------
def to_float_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Return a numeric-only view of the frame with ``NA`` normalised to ``NaN``.

    Keeps all threshold arithmetic in plain ``float64`` (nullable ``Int64``/``boolean``
    columns would otherwise propagate ``pd.NA`` into boolean masks).
    """
    if frame is None or frame.empty:
        return pd.DataFrame()
    work = pd.DataFrame(index=frame.index)
    for spec in TELEMETRY_SCHEMA:
        if spec.name not in frame.columns:
            continue
        if spec.dtype in {"float", "int", "bool"}:
            work[spec.name] = pd.to_numeric(frame[spec.name], errors="coerce").astype("float64")
    if "sensor_fault" in frame.columns:
        work["sensor_fault"] = frame["sensor_fault"].fillna(False).astype(bool)
    return work


def _nullable_to_bool(series: pd.Series) -> pd.Series:
    """Convert a nullable boolean column to a strict bool column (NA -> False)."""
    return series.fillna(False).astype(bool)


# --------------------------------------------------------------------------------------
# Anomaly engine
# --------------------------------------------------------------------------------------
def detect_anomalies(
    frame: pd.DataFrame,
    thresholds: Thresholds = THRESHOLDS,
) -> pd.DataFrame:
    """
    Flag operational anomalies row-by-row.

    Returns a copy of ``frame`` with one boolean column per rule (:data:`FLAG_COLUMNS`),
    plus ``severity``, ``reasons``, ``anomaly_count`` and ``is_anomaly``.

    Robustness contract (exercised by the boundary-value tests)
    ----------------------------------------------------------
    * Works on an empty frame, a frame with ``NaN`` cells, and extreme inputs
      (``rpm = 9999``, ``proximity_m = -1.0``, ``fuel_level_pct = 250``).
    * Impossible values were already nulled by the sanitiser, so here they surface as
      ``flag_sensor_fault`` - the engine never raises, never divides by zero and never
      emits ``inf``.
    """
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        empty = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        for column in FLAG_COLUMNS:
            empty[column] = pd.Series([], dtype=bool)
        empty["severity"] = pd.Series([], dtype="object")
        empty["reasons"] = pd.Series([], dtype="object")
        empty["anomaly_count"] = pd.Series([], dtype="int64")
        empty["is_anomaly"] = pd.Series([], dtype=bool)
        return empty

    out = frame.copy()
    work = to_float_frame(frame)
    index = out.index

    def col(name: str, default: float = math.nan) -> pd.Series:
        """Numeric column accessor that degrades to a constant when absent."""
        if name in work.columns:
            return work[name]
        return pd.Series(default, index=index, dtype="float64")

    rpm = col("rpm")
    speed = col("ground_speed_kph")
    payload = col("payload_tons")
    fuel_pct = col("fuel_level_pct")
    fuel_rate = col("fuel_rate_lph")
    psi = col("hydraulic_psi")
    oil_temp = col("hydraulic_oil_temp_c")
    coolant = col("coolant_temp_c")
    oil_press = col("oil_pressure_kpa")
    proximity = col("proximity_m")
    seatbelt = (
        _nullable_to_bool(frame["seatbelt_latched"])
        if "seatbelt_latched" in frame.columns
        else pd.Series(False, index=index, dtype=bool)
    )

    engine_running = rpm > thresholds.idle_rpm_max
    # "working hard" = moving the machine or carrying material.
    working_hard = (speed > thresholds.idle_speed_kph_max) | (payload > thresholds.idle_payload_tons_max)
    # "hydraulic demand" = the implement circuit is actually loaded (carrying material).
    # Tramming empty at standby pressure is normal, so it must not raise an under-pressure flag.
    hydraulic_demand = payload > thresholds.idle_payload_tons_max

    flags: dict[str, pd.Series] = {}
    flags["flag_over_rev"] = rpm > thresholds.over_rev_rpm
    flags["flag_idle_excess_fuel"] = (rpm <= thresholds.idle_rpm_max) & (
        fuel_rate > thresholds.idle_fuel_rate_lph_max
    )
    flags["flag_low_fuel"] = fuel_pct < thresholds.low_fuel_pct
    flags["flag_critical_fuel"] = fuel_pct < thresholds.critical_fuel_pct
    flags["flag_hydraulic_overpressure"] = psi > thresholds.hydraulic_psi_nominal_high
    flags["flag_hydraulic_underpressure"] = engine_running & hydraulic_demand & (
        psi < thresholds.hydraulic_psi_nominal_low
    )
    flags["flag_hydraulic_overtemp"] = oil_temp > thresholds.hydraulic_oil_temp_c_max
    flags["flag_coolant_overtemp"] = coolant > thresholds.coolant_temp_c_max
    flags["flag_low_oil_pressure"] = engine_running & (oil_press < thresholds.oil_pressure_kpa_min)
    flags["flag_seatbelt_while_moving"] = (
        (speed > thresholds.seatbelt_release_speed_kph) & (~seatbelt)
    )
    flags["flag_proximity_intrusion"] = proximity < thresholds.proximity_warning_m
    flags["flag_proximity_critical"] = proximity < thresholds.proximity_stop_m
    flags["flag_sensor_fault"] = (
        frame["sensor_fault"].fillna(False).astype(bool)
        if "sensor_fault" in frame.columns
        else pd.Series(False, index=index, dtype=bool)
    )

    # A sensor fault invalidates the *derived* conclusions from the same field, so any
    # flag whose input column is NaN is suppressed (an unknown reading is not an event).
    for name, mask in flags.items():
        flags[name] = _nullable_to_bool(mask)

    # Sustained high idle is computed on time windows, then painted back onto rows.
    idle_windows = detect_idle_windows(frame, thresholds)
    idle_mask = pd.Series(False, index=index)
    for window in idle_windows:
        idle_mask |= (frame["timestamp"] >= window.start) & (frame["timestamp"] <= window.end)
    flags["flag_high_idle"] = idle_mask

    for name in FLAG_COLUMNS:
        out[name] = flags.get(name, pd.Series(False, index=index)).astype(bool)

    # --- fold flags into severity / reasons ----------------------------------------
    flag_matrix = pd.DataFrame({name: out[name] for name in FLAG_COLUMNS}, index=index)
    out["anomaly_count"] = flag_matrix.sum(axis=1).astype("int64")
    out["is_anomaly"] = out["anomaly_count"] > 0

    severities: list[str] = []
    reasons: list[str] = []
    for position, (_, row) in enumerate(flag_matrix.iterrows()):
        active = [name for name in FLAG_COLUMNS if bool(row[name])]
        severities.append(_max_severity(active))
        reasons.append("; ".join(FLAG_LABELS[name] for name in active))
    out["severity"] = pd.Series(severities, index=index, dtype="object")
    out["reasons"] = pd.Series(reasons, index=index, dtype="object")

    # NOTE: idle windows are returned by :func:`detect_idle_windows` rather than being
    # attached here, so the frame keeps a rectangular, hash-stable schema.
    return out


def finite_values(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Return only the finite entries of a numeric array.

    Every statistic in this module goes through here: ``np.nanmean``/``np.nanmax`` emit a
    ``RuntimeWarning`` (and return ``nan``) for an all-``NaN`` window, which happens whenever
    a sensor channel is entirely faulty. Filtering first keeps the dashboard warning-free and
    the outputs finite.
    """
    array = np.asarray(values, dtype="float64")
    if array.size == 0:
        return array
    return array[np.isfinite(array)]


def safe_mean(values: Sequence[float] | np.ndarray, default: float = 0.0) -> float:
    """Mean of the finite entries (``default`` when there are none)."""
    finite = finite_values(values)
    return round(float(finite.mean()), 1) if finite.size else float(default)


def safe_max(values: Sequence[float] | np.ndarray, default: float = 0.0) -> float:
    """Maximum of the finite entries (``default`` when there are none)."""
    finite = finite_values(values)
    return round(float(finite.max()), 1) if finite.size else float(default)


def count_events(mask: Sequence[bool] | pd.Series) -> int:
    """
    Count contiguous ``True`` runs in a boolean mask (one run == one event).

    Used by the KPIs so "3 proximity intrusions" means three *occurrences*, not 18 samples.
    Returns 0 for empty/invalid input.
    """
    if mask is None:
        return 0
    series = pd.Series(mask).fillna(False).astype(bool)
    if series.empty:
        return 0
    return int(((series != series.shift(1, fill_value=False)) & series).sum())


def _max_severity(active_flags: Sequence[str]) -> str:
    """Fold a set of active flags into the highest matching severity."""
    level = "OK"
    for name in active_flags:
        severity = FLAG_SEVERITY.get(name, "INFO")
        if SEVERITY_ORDER.index(severity) > SEVERITY_ORDER.index(level):
            level = severity
    return level


def detect_idle_windows(
    frame: pd.DataFrame,
    thresholds: Thresholds = THRESHOLDS,
) -> list[IdleWindow]:
    """
    Find *sustained* idle periods (engine running, no travel, no payload).

    A window is only reported when it lasts at least ``thresholds.high_idle_seconds``,
    which is what separates normal "waiting for the truck" from actionable fuel waste.
    Returns an empty list for empty/invalid input.
    """
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    if "timestamp" not in frame.columns:
        return []

    work = to_float_frame(frame)
    if work.empty:
        return []

    idle_now = (
        (work["rpm"] <= thresholds.idle_rpm_max)
        & (work["ground_speed_kph"] <= thresholds.idle_speed_kph_max)
        & (work["payload_tons"] <= thresholds.idle_payload_tons_max)
    ).fillna(False).astype(bool)

    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    interval = float(thresholds.sample_interval_s)

    windows: list[IdleWindow] = []
    group_id = (idle_now != idle_now.shift(1, fill_value=False)).cumsum()
    for _, group in pd.DataFrame(
        {"idle": idle_now.to_numpy(), "ts": timestamps.to_numpy(), "rpm": work["rpm"].to_numpy(),
         "fuel": work["fuel_rate_lph"].to_numpy(), "g": group_id.to_numpy()},
        index=frame.index,
    ).groupby("g", sort=True):
        if not bool(group["idle"].iloc[0]):
            continue
        start = pd.Timestamp(group["ts"].iloc[0])
        end = pd.Timestamp(group["ts"].iloc[-1])
        duration = float((end - start).total_seconds()) + interval
        if duration >= thresholds.high_idle_seconds:
            fuel_values = finite_values(group["fuel"].to_numpy(dtype="float64"))
            fuel = float(fuel_values.sum()) * interval / 3600.0 if fuel_values.size else 0.0
            windows.append(
                IdleWindow(
                    start=start,
                    end=end,
                    duration_s=round(duration, 1),
                    avg_rpm=safe_mean(group["rpm"].to_numpy(dtype="float64")),
                    fuel_burned_l=round(fuel, 2),
                )
            )
    return windows


def detect_haul_cycles(
    frame: pd.DataFrame,
    thresholds: Thresholds = THRESHOLDS,
) -> list[HaulCycle]:
    """
    Segment payload telemetry into haul cycles.

    A cycle is a contiguous run of loaded samples (``payload >= min_payload_for_cycle_tons``);
    its payload is the **peak** of the run, which avoids double-counting the ramp up/down of
    the same bucket load.
    """
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return []

    work = to_float_frame(frame)
    if work.empty or "payload_tons" not in work.columns or "timestamp" not in frame.columns:
        return []

    loaded = (work["payload_tons"] >= thresholds.min_payload_for_cycle_tons).fillna(False).astype(bool)
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    interval = float(thresholds.sample_interval_s)
    group_id = (loaded != loaded.shift(1, fill_value=False)).cumsum()

    cycles: list[HaulCycle] = []
    for _, group in pd.DataFrame(
        {"loaded": loaded.to_numpy(), "ts": timestamps.to_numpy(),
         "payload": work["payload_tons"].to_numpy(), "g": group_id.to_numpy()},
        index=frame.index,
    ).groupby("g", sort=True):
        if not bool(group["loaded"].iloc[0]):
            continue
        start = pd.Timestamp(group["ts"].iloc[0])
        end = pd.Timestamp(group["ts"].iloc[-1])
        payloads = finite_values(group["payload"].to_numpy(dtype="float64"))
        cycles.append(
            HaulCycle(
                start=start,
                end=end,
                peak_payload_tons=round(float(payloads.max()), 2) if payloads.size else 0.0,
                duration_s=round(float((end - start).total_seconds()) + interval, 1),
            )
        )
    return cycles


# --------------------------------------------------------------------------------------
# KPIs + machine status
# --------------------------------------------------------------------------------------
def compute_kpis(
    frame: pd.DataFrame,
    thresholds: Thresholds = THRESHOLDS,
) -> KpiSummary:
    """
    Compute the dashboard KPI block from a sanitised frame.

    Always returns a :class:`KpiSummary`; an empty frame yields an all-zero "NO_DATA" KPI.
    Integrals use a clipped sampling interval so a clock jump cannot fabricate fuel burn.
    """
    kpi = KpiSummary()
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return kpi

    kpi.rows = int(len(frame))
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    valid_ts = timestamps.dropna()
    if not valid_ts.empty:
        kpi.window_start = valid_ts.iloc[0]
        kpi.window_end = valid_ts.iloc[-1]
        kpi.window_hours = max(
            0.0, float((valid_ts.iloc[-1] - valid_ts.iloc[0]).total_seconds()) / 3600.0
        )

    work = to_float_frame(frame)
    interval = float(thresholds.sample_interval_s)

    cycles = detect_haul_cycles(frame, thresholds)
    kpi.cycles = len(cycles)
    kpi.tons_hauled = round(float(sum(c.peak_payload_tons for c in cycles)), 2)
    kpi.avg_cycle_s = round(float(np.mean([c.duration_s for c in cycles])), 1) if cycles else 0.0

    # Integration window: use real deltas when usable, clipped to a sane range.
    if len(valid_ts) >= 2:
        deltas = timestamps.diff().dt.total_seconds().to_numpy(dtype="float64")
        deltas = np.where(np.isfinite(deltas) & (deltas > 0) & (deltas <= 3600), deltas, np.nan)
        fallback = float(np.nanmedian(deltas)) if np.isfinite(np.nanmedian(deltas)) else interval
        deltas = np.where(np.isnan(deltas), fallback, deltas)
    else:
        deltas = np.full(len(frame), interval, dtype="float64")

    if "fuel_rate_lph" in work.columns:
        rates = work["fuel_rate_lph"].to_numpy(dtype="float64")
        rates = np.where(np.isfinite(rates), rates, 0.0)
        kpi.fuel_burned_l = round(float(np.sum(rates * deltas / 3600.0)), 2)

    if "rpm" in work.columns and kpi.window_hours > 0:
        rpm_values = work["rpm"].to_numpy(dtype="float64")
        idle = (
            (work["rpm"] <= thresholds.idle_rpm_max)
            & (work["ground_speed_kph"] <= thresholds.idle_speed_kph_max)
            & (work["payload_tons"] <= thresholds.idle_payload_tons_max)
        ).fillna(False).to_numpy(dtype=bool)
        kpi.idle_pct = round(float(np.sum(idle) / max(1, len(idle)) * 100.0), 1)
        kpi.avg_rpm = safe_mean(rpm_values)
        kpi.max_rpm = safe_max(rpm_values)

    if "hydraulic_psi" in work.columns:
        psi_values = work["hydraulic_psi"].to_numpy(dtype="float64")
        psi_values = finite_values(psi_values)
        kpi.p95_hydraulic_psi = round(float(np.percentile(psi_values, 95)), 1) if len(psi_values) else 0.0

    kpi.tons_per_hour = (
        round(kpi.tons_hauled / kpi.window_hours, 2) if kpi.window_hours > 0 else 0.0
    )
    kpi.fuel_per_ton_l = (
        round(kpi.fuel_burned_l / kpi.tons_hauled, 2) if kpi.tons_hauled > 0 else 0.0
    )

    if "is_anomaly" in frame.columns:
        kpi.anomaly_rows = int(frame["is_anomaly"].fillna(False).astype(bool).sum())
        event_flags = (
            "flag_over_rev",
            "flag_proximity_intrusion",
            "flag_seatbelt_while_moving",
            "flag_hydraulic_overpressure",
            "flag_low_oil_pressure",
        )
        kpi.anomaly_events = len(detect_idle_windows(frame, thresholds)) + sum(
            count_events(frame[flag]) for flag in event_flags if flag in frame.columns
        )
        kpi.critical_events = int(
            (frame.get("severity", pd.Series(dtype="object")).astype(str) == "CRITICAL").sum()
        )

    kpi.status = machine_status(frame, thresholds).split(" ")[0]
    return kpi


def machine_status(frame: pd.DataFrame, thresholds: Thresholds = THRESHOLDS) -> str:
    """
    Summarise the machine state as ``"RUNNING <detail>"`` etc.

    Priority: ``NO_DATA`` -> ``FAULT`` -> ``STOPPED`` -> ``IDLE`` -> ``RUNNING``.
    """
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return "NO_DATA (no telemetry)"

    work = to_float_frame(frame)
    if work.empty:
        return "NO_DATA (no numeric telemetry)"

    last = work.iloc[-1]
    rpm = float(last.get("rpm", math.nan))
    speed = float(last.get("ground_speed_kph", math.nan))
    payload = float(last.get("payload_tons", math.nan))

    if frame["sensor_fault"].fillna(False).astype(bool).iloc[-1] and not math.isfinite(rpm):
        return "FAULT (last sample invalid)"
    if not math.isfinite(rpm):
        return "NO_DATA (rpm unavailable)"
    if rpm == 0:
        return "STOPPED (engine off)"
    if rpm <= thresholds.idle_rpm_max and speed <= thresholds.idle_speed_kph_max and payload <= thresholds.idle_payload_tons_max:
        return "IDLE (engine on, no work)"
    if payload > thresholds.idle_payload_tons_max:
        return "RUNNING (hauling)"
    if speed > thresholds.idle_speed_kph_max:
        return "RUNNING (tramming)"
    return "RUNNING"


# --------------------------------------------------------------------------------------
# Aggregation helpers for the UI
# --------------------------------------------------------------------------------------
def summarize_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``flag · label · count · severity`` rows for the anomaly summary chart."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=["flag", "label", "count", "severity"])
    rows = []
    for name in FLAG_COLUMNS:
        if name in frame.columns:
            count = int(frame[name].fillna(False).astype(bool).sum())
            if count:
                rows.append(
                    {
                        "flag": name,
                        "label": FLAG_LABELS.get(name, name),
                        "count": count,
                        "severity": FLAG_SEVERITY.get(name, "INFO"),
                    }
                )
    summary = pd.DataFrame(rows, columns=["flag", "label", "count", "severity"])
    if not summary.empty:
        summary = summary.sort_values("count", ascending=False).reset_index(drop=True)
    return summary


def anomaly_log(frame: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    """
    Build the operator-facing anomaly log.

    Every field is produced from static labels plus numeric casts - **no raw ingestion
    string is ever echoed**, which removes the stored-XSS surface from the log view.
    """
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=["timestamp", "severity", "reasons", "machine_id"])
    if "is_anomaly" not in frame.columns:
        return pd.DataFrame(columns=["timestamp", "severity", "reasons", "machine_id"])
    mask = frame["is_anomaly"].fillna(False).astype(bool)
    events = frame.loc[mask, ["timestamp", "severity", "reasons"]].copy()
    if events.empty:
        return pd.DataFrame(columns=["timestamp", "severity", "reasons", "machine_id"])
    events["machine_id"] = [
        flatten_for_log(value, 32) for value in frame.loc[mask, "machine_id"].fillna("UNKNOWN")
    ]
    events = events.sort_values("timestamp", ascending=False).head(max(1, int(limit)))
    return events.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Simulator (offline demo + test fixture source)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SimulationConfig:
    """Knobs for :func:`simulate_telemetry` (all deterministic given ``seed``)."""

    machine_id: str = "CAT-320-EXC-014"
    minutes: float = 120.0
    interval_s: float = 5.0
    seed: int = 20260923
    cycle_period_s: float = 168.0
    inject_over_rev: bool = True
    inject_high_idle: bool = True
    inject_proximity_intrusion: bool = True
    inject_seatbelt_release: bool = True
    inject_low_fuel: bool = True
    inject_sensor_fault: bool = False
    start: datetime | None = None


def simulate_telemetry(config: SimulationConfig | None = None, **overrides: Any) -> pd.DataFrame:
    """
    Generate a realistic, deterministic telemetry stream for one shift window.

    The generated profile contains a proper haul cycle (dig ▸ swing ▸ dump ▸ return),
    realistic RPM/hydraulic/fuel coupling, and - unless disabled - a scripted set of
    faults so the Safety Guardian and the anomaly engine have something to find:

    ``inject_over_rev`` (2 400 rpm spike), ``inject_high_idle`` (8 min idle),
    ``inject_proximity_intrusion`` (3.5 m pedestrian), ``inject_seatbelt_release``
    (unlatched while tramming) and ``inject_low_fuel`` (tank draining to 6 %).

    Uses ``numpy.random.default_rng`` (PCG64) - **not** the ``random`` module - so the
    stream is reproducible and the codebase stays bandit ``B311`` clean.
    """
    cfg = config or SimulationConfig(**overrides)
    interval = float(cfg.interval_s)
    n = max(2, int(round(cfg.minutes * 60.0 / interval)))
    rng = np.random.default_rng(cfg.seed)
    start = cfg.start or datetime(2026, 9, 23, 6, 0, 0)
    timestamps = [start + timedelta(seconds=interval * i) for i in range(n)]
    t = np.arange(n, dtype="float64") * interval

    phase = (t % cfg.cycle_period_s) / cfg.cycle_period_s
    digging = (phase >= 0.10) & (phase < 0.45)
    swinging = (phase >= 0.45) & (phase < 0.75)
    dumping = (phase >= 0.75) & (phase < 0.90)

    payload = np.where(digging, 9.4 * (0.55 + 0.45 * np.sin(np.pi * (phase - 0.10) / 0.35)), 0.0)
    payload = np.where(swinging, 9.4 * (0.9 + 0.1 * rng.random(n)), payload)
    payload = np.where(dumping, np.clip(payload * 0.4, 0.0, None), payload)
    payload += rng.normal(0.0, 0.25, n).clip(-0.4, 0.4)
    payload = np.clip(payload, 0.0, 12.5)

    loading = payload > 0.5
    rpm = np.where(loading, 1_760.0, np.where(phase >= 0.90, 1_250.0, 880.0))
    rpm += rng.normal(0.0, 22.0, n)
    rpm = np.clip(rpm, 600.0, 2_050.0)

    ground_speed = np.where(loading, 1.6, np.where(phase >= 0.90, 7.5, 0.0))
    ground_speed += np.abs(rng.normal(0.0, 0.4, n))

    hydraulic_psi = np.where(loading, 4_250.0, 620.0) + rng.normal(0.0, 180.0, n)
    hydraulic_psi = np.clip(hydraulic_psi, 200.0, 6_400.0)

    hydraulic_temp = 62.0 + 18.0 * (t / max(1.0, t[-1])) + rng.normal(0.0, 1.1, n)
    coolant_temp = 84.0 + 6.0 * (t / max(1.0, t[-1])) + rng.normal(0.0, 1.0, n)
    oil_pressure = np.where(rpm > 1_000, 380.0, 220.0) + rng.normal(0.0, 18.0, n)

    fuel_rate = np.where(loading, 32.0, 6.4) + rng.normal(0.0, 1.1, n)
    fuel_rate = np.clip(fuel_rate, 3.0, 60.0)

    proximity = np.full(n, 24.0) + rng.normal(0.0, 1.2, n)
    proximity = np.clip(proximity, 0.5, 30.0)

    seatbelt = np.ones(n, dtype=bool)

    # --- scripted, deterministic faults -------------------------------------------
    if cfg.inject_high_idle:
        lo, hi = int(n * 0.30), int(n * 0.30) + int((THRESHOLDS.high_idle_seconds + 300) / interval)
        hi = min(hi, n - 1)
        rpm[lo:hi] = 860.0 + rng.normal(0.0, 12.0, hi - lo)
        ground_speed[lo:hi] = 0.0
        payload[lo:hi] = 0.0
        hydraulic_psi[lo:hi] = 430.0 + rng.normal(0.0, 60.0, hi - lo)
        fuel_rate[lo:hi] = 6.5 + rng.normal(0.0, 0.5, hi - lo)

    if cfg.inject_over_rev:
        spike = min(n - 3, int(n * 0.55))
        rpm[spike : spike + 2] = 2_400.0
        hydraulic_psi[spike : spike + 2] = 5_350.0

    if cfg.inject_proximity_intrusion:
        near = min(n - 6, int(n * 0.72))
        proximity[near : near + 6] = 3.4
        ground_speed[near : near + 6] = 0.5

    if cfg.inject_seatbelt_release:
        unbuckled = min(n - 8, int(n * 0.86))
        seatbelt[unbuckled : unbuckled + 8] = False
        ground_speed[unbuckled : unbuckled + 8] = 8.0

    if cfg.inject_low_fuel:
        fuel_level = np.linspace(92.0, 26.0, n)
        tail = max(1, int(n * 0.05))
        fuel_level[-tail:] = np.linspace(13.0, 6.0, tail)
    else:
        fuel_level = np.linspace(92.0, 34.0, n)

    if cfg.inject_sensor_fault:
        broken = max(0, n // 2)
        rpm[broken] = np.nan
        proximity[broken] = -1.0

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "machine_id": cfg.machine_id,
            "rpm": np.rint(rpm),
            "fuel_rate_lph": np.round(fuel_rate, 2),
            "fuel_level_pct": np.round(fuel_level, 2),
            "hydraulic_psi": np.round(hydraulic_psi, 1),
            "hydraulic_oil_temp_c": np.round(hydraulic_temp, 1),
            "coolant_temp_c": np.round(coolant_temp, 1),
            "oil_pressure_kpa": np.round(oil_pressure, 1),
            "ground_speed_kph": np.round(ground_speed, 2),
            "proximity_m": np.round(proximity, 2),
            "payload_tons": np.round(payload, 2),
            "seatbelt_latched": seatbelt,
        }
    )
    return frame


def telemetry_to_csv(frame: pd.DataFrame, path: str | Path | None = None) -> str:
    """Serialise a frame to CSV (returns the text; optionally writes it to ``path``)."""
    text = frame.to_csv(index=False)
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return text


__all__ = [
    "FLAG_COLUMNS",
    "FLAG_LABELS",
    "FLAG_SEVERITY",
    "TelemetryIngestResult",
    "IdleWindow",
    "HaulCycle",
    "KpiSummary",
    "SimulationConfig",
    "empty_telemetry_frame",
    "load_telemetry_csv",
    "sanitize_telemetry_frame",
    "to_float_frame",
    "detect_anomalies",
    "detect_idle_windows",
    "detect_haul_cycles",
    "count_events",
    "finite_values",
    "safe_mean",
    "safe_max",
    "compute_kpis",
    "machine_status",
    "summarize_flags",
    "anomaly_log",
    "simulate_telemetry",
    "telemetry_to_csv",
]
