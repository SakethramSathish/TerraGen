#!/usr/bin/env python3
"""
scripts/generate_sample_data.py
===============================
Regenerate ``data/sample_telemetry.csv`` - a 3-hour shift window from the edge simulator.

Why ship a CSV at all?
----------------------
The bundled file gives a **zero-setup demo** (``Telemetry source ▸ Bundled sample CSV``)
and doubles as a golden fixture for regression tests: it is deterministic, so a change in
the ingest/anomaly logic shows up as a diff instead of a surprise in the field.

Usage::

    python scripts/generate_sample_data.py            # 180 min, default seed
    python scripts/generate_sample_data.py --minutes 60 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Allow ``python scripts/generate_sample_data.py`` from the project root without install.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import telemetry  # noqa: E402  (path bootstrap must run first)
from config import DEFAULT_MACHINE_ID, DEFAULT_SIM_SEED, SAMPLE_CSV  # noqa: E402


def main() -> int:
    """Generate the sample CSV and print a short provenance summary."""
    parser = argparse.ArgumentParser(description="Generate the bundled demo telemetry CSV.")
    parser.add_argument("--minutes", type=float, default=180.0, help="window length (default 180)")
    parser.add_argument("--interval", type=float, default=5.0, help="sample interval seconds")
    parser.add_argument("--seed", type=int, default=DEFAULT_SIM_SEED, help="RNG seed")
    parser.add_argument("--machine", default=DEFAULT_MACHINE_ID, help="machine id")
    parser.add_argument("--start", default="2026-09-23T06:00:00", help="window start (ISO-8601)")
    parser.add_argument("--output", type=Path, default=SAMPLE_CSV, help="output path")
    parser.add_argument("--fleet", action="store_true", default=True, help="generate multi-machine fleet dataset")
    args = parser.parse_args()

    fleet_ids = (
        ("CAT-320-EXC-014", int(args.seed)),
        ("CAT-745-ART-002", int(args.seed) + 10),
        ("CAT-D6-DOZ-008", int(args.seed) + 20),
        ("CAT-980-WLD-005", int(args.seed) + 30),
    ) if args.fleet and args.machine == DEFAULT_MACHINE_ID else ((args.machine, int(args.seed)),)

    import pandas as pd
    frames = []
    for mid, m_seed in fleet_ids:
        cfg = telemetry.SimulationConfig(
            machine_id=mid,
            minutes=float(args.minutes),
            interval_s=float(args.interval),
            seed=m_seed,
            start=datetime.fromisoformat(args.start),
        )
        frames.append(telemetry.simulate_telemetry(cfg))

    combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    # A sample file is *input* data: push it through the production sanitiser and write the
    # sanitised frame, so the shipped CSV is guaranteed ingestible.
    sanitized = telemetry.sanitize_telemetry_frame(combined).frame
    telemetry.telemetry_to_csv(sanitized, args.output)

    anomalies = telemetry.detect_anomalies(sanitized)
    print(f"wrote {args.output} ({len(sanitized):,} rows, machines: {list(sanitized['machine_id'].unique())})")
    print(f"  window      : {sanitized['timestamp'].iloc[0]} -> {sanitized['timestamp'].iloc[-1]}")
    print(f"  flagged rows: {int(anomalies['is_anomaly'].sum()):,}")
    print(f"  digest      : {telemetry.sanitize_telemetry_frame(combined).digest[:32]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
