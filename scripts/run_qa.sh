#!/usr/bin/env bash
# =====================================================================================
# scripts/run_qa.sh - multi-layered quality & security gate for the
#                     CAT Smart Operator Assistant
# -------------------------------------------------------------------------------------
# Run this before every deployment (and in CI). It exits non-zero on the first failure,
# so it can be wired straight into a pipeline:
#
#     bash scripts/run_qa.sh
#
# Layers
#   0. environment      Python version + dependency check
#   1. syntax           byte-compile every module (catches import-time typos)
#   2. unit + integration tests   pytest (telemetry, gatekeeper, XSS, dashboard, safety, UI)
#   3. SAST              bandit static analysis (must report zero issues)
#   4. artifact check    sample telemetry regenerates and ingests cleanly
# =====================================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

BLUE="\033[1;34m"; GREEN="\033[1;32m"; RED="\033[1;31m"; NC="\033[0m"
step() { printf "${BLUE}==> %s${NC}\n" "$1"; }
pass() { printf "${GREEN}    ✓ %s${NC}\n" "$1"; }
fail() { printf "${RED}    ✗ %s${NC}\n" "$1"; exit 1; }

PYTHON="${PYTHON:-python3}"

step "0/4 Environment"
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), sys.version; print(f"    python {sys.version.split()[0]}")'
"$PYTHON" -c 'import streamlit, pandas, numpy, pytest' || fail "dependencies missing - run: pip install -r requirements.txt"
pass "dependencies present (streamlit, pandas, numpy, pytest)"

step "1/4 Syntax / import check"
"$PYTHON" -m compileall -q app.py config.py core scripts >/dev/null || fail "byte-compile failed"
"$PYTHON" -c 'import app, config; from core import copilot, dashboard, safety, security, telemetry, audit, llm_gateway, knowledge_base, formatting' \
  || fail "module import failed"
pass "all modules compile and import"

step "2/4 Unit, security and integration tests (pytest)"
"$PYTHON" -m pytest tests/ -q || fail "test suite failed"
pass "test suite green"

step "3/4 SAST (bandit)"
if command -v bandit >/dev/null 2>&1; then
  BANDIT_OUT="$(bandit -r app.py config.py core scripts -f txt 2>&1 || true)"
  echo "$BANDIT_OUT" | sed -n '/Test results/,/Files skipped/p'
  echo "$BANDIT_OUT" | grep -q "No issues identified" || fail "bandit reported issues"
  pass "bandit: no issues identified"
else
  printf "    ! bandit not installed - skipping (pip install bandit)\n"
fi

step "4/4 Artifact / pipeline check"
"$PYTHON" - <<'PY' || fail "telemetry pipeline check failed"
import sys
sys.path.insert(0, ".")
from core import telemetry

frame = telemetry.simulate_telemetry(telemetry.SimulationConfig(minutes=30))
result = telemetry.sanitize_telemetry_frame(frame)
assert result.rows_kept > 0, "no rows ingested"
analyse = telemetry.detect_anomalies(result.frame)
kpis = telemetry.compute_kpis(analyse)
print(f"    rows={result.rows_kept} anomalies={kpis.anomaly_rows} "
      f"cycles={kpis.cycles} tons={kpis.tons_hauled} digest={result.digest[:12]}")
PY
pass "telemetry pipeline produces analysable data"

printf "${GREEN}==> QA gate passed - safe to deploy.${NC}\n"
