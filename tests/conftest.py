"""
tests/conftest.py
=================
Shared pytest fixtures and helpers for the CAT Smart Operator Assistant suite.

Test-isolation decisions
------------------------
* The project root is put on ``sys.path`` so ``import app`` / ``import core`` work from the
  tests directory without an editable install (``pytest tests/`` and ``pytest`` both work).
* ``CAT_DB_PATH`` is redirected to a throw-away directory **before** ``config`` is imported,
  so the suite never touches the operator's real audit trail.
* Telemetry fixtures are session-scoped: the simulator is deterministic, so building the
  standard frame once keeps the suite fast enough to run before every deployment.
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile
from typing import Any, Sequence

# --- path + isolation bootstrap (must run before any project import) -------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEST_ARTIFACTS = pathlib.Path(tempfile.mkdtemp(prefix="cat-assistant-tests-"))
os.environ.setdefault("CAT_DB_PATH", str(TEST_ARTIFACTS / "audit.sqlite3"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from core import telemetry  # noqa: E402
from core.llm_gateway import LLMResult  # noqa: E402

MACHINE_ID = "CAT-320-EXC-014"

#: Canonical column order, mirroring ``config.TELEMETRY_SCHEMA``.
COLUMNS: tuple[str, ...] = (
    "timestamp", "machine_id", "rpm", "fuel_rate_lph", "fuel_level_pct", "hydraulic_psi",
    "hydraulic_oil_temp_c", "coolant_temp_c", "oil_pressure_kpa", "ground_speed_kph",
    "proximity_m", "payload_tons", "seatbelt_latched",
)

#: A healthy, productive working row - every test perturbs exactly what it is testing.
HEALTHY_ROW: dict[str, Any] = {
    "rpm": 1_800,
    "fuel_rate_lph": 30.0,
    "fuel_level_pct": 70.0,
    "hydraulic_psi": 4_200.0,
    "hydraulic_oil_temp_c": 70.0,
    "coolant_temp_c": 88.0,
    "oil_pressure_kpa": 380.0,
    "ground_speed_kph": 3.0,
    "proximity_m": 20.0,
    "payload_tons": 6.0,
    "seatbelt_latched": True,
}


def build_frame(
    rows: int = 4,
    *,
    start: str = "2026-09-23 06:00:00",
    freq: str = "5s",
    **overrides: Any,
) -> pd.DataFrame:
    """
    Build a telemetry frame.

    Each override accepts either a scalar (broadcast to every row) or a sequence of exactly
    ``rows`` values. Example::

        build_frame(3, rpm=[900, 900, 2_400], payload_tons=0.0)
    """
    data: dict[str, Any] = {}
    for key, value in {**HEALTHY_ROW, **overrides}.items():
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            sequence = list(value)
            if len(sequence) != rows:
                raise ValueError(f"override '{key}' has {len(sequence)} values, expected {rows}")
            data[key] = sequence
        else:
            data[key] = [value] * rows
    frame = pd.DataFrame(data)
    frame["timestamp"] = pd.date_range(start, periods=rows, freq=freq)
    frame["machine_id"] = MACHINE_ID
    return frame[list(COLUMNS)]


class LLMSpy:
    """
    Callable stand-in for the remote LLM.

    Records every message list it is handed, so a test can assert the security property
    "out-of-scope / injected queries never reach a model" with a hard count of zero.
    """

    def __init__(self, text: str = "Hydraulic pressure below 1500 psi while loaded means check "
                                   "the suction strainer and oil level.",
                 ok: bool = True, error: str = "") -> None:
        self.text = text
        self.ok = ok
        self.error = error
        self.calls: list[list[dict[str, str]]] = []

    def __call__(self, messages: Sequence[dict[str, str]]) -> LLMResult:
        self.calls.append([dict(m) for m in messages])
        if not self.ok:
            return LLMResult(ok=False, error=self.error or "spy failure", model="spy-model")
        return LLMResult(ok=True, text=self.text, model="spy-model", latency_ms=1.0)

    @property
    def call_count(self) -> int:
        """How many times the model was invoked."""
        return len(self.calls)


class FakeClock:
    """Deterministic monotonic clock for rate-limiter tests (no ``sleep``)."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward."""
        self.now += float(seconds)


@pytest.fixture(scope="session")
def sim_frame() -> pd.DataFrame:
    """A 45-minute simulated shift window (deterministic)."""
    return telemetry.simulate_telemetry(telemetry.SimulationConfig(minutes=45))


@pytest.fixture(scope="session")
def clean_frame(sim_frame: pd.DataFrame) -> pd.DataFrame:
    """The simulated window after the production ingest sanitiser."""
    return telemetry.sanitize_telemetry_frame(sim_frame).frame


@pytest.fixture(scope="session")
def analysed(clean_frame: pd.DataFrame) -> pd.DataFrame:
    """The sanitised window after the anomaly engine."""
    return telemetry.detect_anomalies(clean_frame)


@pytest.fixture
def frame_factory():
    """Expose :func:`build_frame` as a fixture."""
    return build_frame


@pytest.fixture
def null_frame() -> pd.DataFrame:
    """A correctly-typed, zero-row frame (the "no data" UI path)."""
    return telemetry.empty_telemetry_frame()


@pytest.fixture
def llm_spy() -> LLMSpy:
    """Counting LLM stand-in."""
    return LLMSpy()


@pytest.fixture
def fake_clock() -> FakeClock:
    """Injected monotonic clock."""
    return FakeClock()
