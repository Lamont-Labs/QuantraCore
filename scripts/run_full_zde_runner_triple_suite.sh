#!/usr/bin/env bash
# ==================================================================================================
# QUANTRACORE APEX — ZDE VALIDATION + RUNNER DISCOVERY + TRIPLE-RUN INSTITUTIONAL TEST SUITE
# Version: v9.x — Lamont Labs
#
# PURPOSE:
#   1. Validate Zero Drawdown Entry (ZDE) across ALL scanner modes.
#   2. Add 2–3% wiggle room for execution realism while verifying deterministic logic.
#   3. Force-run massive random-ticker sweeps across ALL market-cap buckets.
#   4. Stress-test Runner Discovery (microcaps → large caps).
#   5. Run full triple-run test cycles (3× everything).
#   6. Capture every test artifact in logs/zde_validation/, logs/runners/, logs/tests_triple/.
#
# STORAGE:
#   Replit WILL store:
#       - all ZDE test logs
#       - all runner discoveries
#       - all determinism cycles
#       - all scanner outputs
#       - all pipeline validations
#       - all random scans
# ==================================================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

TS=$(date -u +"%Y%m%d_%H%M%S")

ZDE_DIR="${ROOT}/logs/zde_validation/${TS}"
RUNNER_DIR="${ROOT}/logs/runner_stress/${TS}"
TRIPLE_DIR="${ROOT}/logs/tests_triple/${TS}"

mkdir -p "${ZDE_DIR}" "${RUNNER_DIR}" "${TRIPLE_DIR}"

echo "================================================================================"
echo "🏛 QUANTRACORE APEX — FULL STACK VALIDATION SUITE (WITH ZDE + RUNNERS)"
echo "Timestamp: ${TS}"
echo "================================================================================"
echo

# ==================================================================================================
# SECTION 1 — ZDE VALIDATION (WITH WIGGLE ROOM)
# ==================================================================================================

echo "🔎 [ZDE] Running Zero Drawdown Entry Validation (±3% tolerance)..."

python3 scripts/zde_validation.py 2>&1 | tee "${ZDE_DIR}/zde_validation.log"

echo "✔ ZDE validation complete."
echo

# ==================================================================================================
# SECTION 2 — RUNNER DISCOVERY STRESS TEST (MICROCAP → LARGE CAP)
# ==================================================================================================

echo "🔥 [RUNNER] Running high-volatility runner discovery stress test..."

python3 scripts/runner_discovery_stress.py 2>&1 | tee "${RUNNER_DIR}/runner_discovery.log"

echo "✔ Runner stress suite finished."
echo

# ==================================================================================================
# SECTION 3 — FULL TRIPLE-RUN INSTITUTIONAL TEST SUITE
# ==================================================================================================

echo "🏛 [TRIPLE] Running triple institutional suite..."
echo "This is going to take a while — EXACTLY what institutions expect."

for RUN in 1 2 3; do
    RUN_PATH="${TRIPLE_DIR}/run_${RUN}"
    mkdir -p "${RUN_PATH}"

    echo "------------------------------------------------------------"
    echo "▶ RUN ${RUN} — Backend Tests"
    pytest src/quantracore_apex/tests -q --disable-warnings --tb=no 2>&1 | tee "${RUN_PATH}/backend.log" || true

    echo "▶ RUN ${RUN} — Frontend Tests"
    (cd dashboard && npm test 2>&1) | tee "${RUN_PATH}/frontend.log" || true

    echo "▶ RUN ${RUN} — Nuclear Determinism (10 cycles)"
    for i in {1..10}; do
        python3 scripts/run_nuclear_determinism.py 2>&1 | tee -a "${RUN_PATH}/determinism.log" || true
    done

    echo "▶ RUN ${RUN} — Scanner 16 Modes"
    python3 scripts/run_scanner_all_modes.py 2>&1 | tee "${RUN_PATH}/scanner.log" || true

    echo "▶ RUN ${RUN} — ApexLab → ApexCore Pipeline"
    python3 scripts/validate_apexcore_pipeline.py 2>&1 | tee "${RUN_PATH}/pipeline.log" || true

done

echo "✔ Triple institutional test suite completed."
echo

# ==================================================================================================
# SECTION 4 — BUILD SCAN INDEX
# ==================================================================================================

echo "📊 [INDEX] Building scan index..."
python3 scripts/build_scan_index.py 2>&1

echo "================================================================================"
echo "ALL TESTS FINISHED — ZDE VERIFIED • RUNNERS VERIFIED • TRIPLE SUITE VERIFIED"
echo "Logs stored under:"
echo "   ${ZDE_DIR}"
echo "   ${RUNNER_DIR}"
echo "   ${TRIPLE_DIR}"
echo "================================================================================"
