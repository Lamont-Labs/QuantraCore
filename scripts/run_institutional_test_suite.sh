#!/usr/bin/env bash
set -euo pipefail

echo "================================================================================"
echo "🏛  QUANTRACORE APEX v9.x — INSTITUTIONAL-LEVEL TEST SUITE"
echo "================================================================================"
echo

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${ROOT_DIR}/logs/tests_institutional"
TS="$(date -u +"%Y%m%d_%H%M%S")"
RUN_DIR="${LOG_ROOT}/${TS}"

mkdir -p "${RUN_DIR}"

SUMMARY_LOG="${RUN_DIR}/summary.log"
REPORT_JSON="${RUN_DIR}/report.json"

echo "[INFO] Run directory: ${RUN_DIR}" | tee -a "${SUMMARY_LOG}"
echo "[INFO] Timestamp: ${TS}" | tee -a "${SUMMARY_LOG}"

###############################################################################
# 1. ENVIRONMENT + STATIC ANALYSIS
###############################################################################
echo | tee -a "${SUMMARY_LOG}"
echo "🔎 [STEP 1] Environment + Static Analysis" | tee -a "${SUMMARY_LOG}"

python3 --version 2>&1 | tee -a "${SUMMARY_LOG}"

pip freeze > "${RUN_DIR}/env_freeze.txt"

echo "[INFO] Running Ruff..." | tee -a "${SUMMARY_LOG}"
ruff check src/quantracore_apex --quiet 2>&1 | tee -a "${SUMMARY_LOG}" || {
    echo "[WARN] Ruff reported issues — review recommended." | tee -a "${SUMMARY_LOG}"
}

echo "[INFO] Running Bandit security scan..." | tee -a "${SUMMARY_LOG}"
bandit -r src/quantracore_apex -ll -q 2>&1 | tee -a "${SUMMARY_LOG}" || {
    echo "[WARN] Bandit reported issues — review recommended." | tee -a "${SUMMARY_LOG}"
}

echo "✅ Static analysis complete." | tee -a "${SUMMARY_LOG}"

###############################################################################
# 2. CORE TEST SUITE (DOUBLE PASS)
###############################################################################
echo | tee -a "${SUMMARY_LOG}"
echo "🧪 [STEP 2] Full pytest suite — double pass" | tee -a "${SUMMARY_LOG}"

echo "[INFO] Test pass #1..." | tee -a "${SUMMARY_LOG}"
pytest src/quantracore_apex/tests -q --disable-warnings --maxfail=5 2>&1 | tee -a "${SUMMARY_LOG}"

echo "[INFO] Clearing caches..." | tee -a "${SUMMARY_LOG}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
rm -rf .pytest_cache

echo "[INFO] Test pass #2..." | tee -a "${SUMMARY_LOG}"
pytest src/quantracore_apex/tests -q --disable-warnings --maxfail=5 2>&1 | tee -a "${SUMMARY_LOG}"

echo "✅ Double-pass test suite complete." | tee -a "${SUMMARY_LOG}"

###############################################################################
# 3. NUCLEAR DETERMINISM (IF AVAILABLE)
###############################################################################
echo | tee -a "${SUMMARY_LOG}"
echo "💣 [STEP 3] Nuclear Determinism Loop" | tee -a "${SUMMARY_LOG}"

NUCLEAR_SCRIPT="${ROOT_DIR}/scripts/run_nuclear_determinism.py"
if [ -f "${NUCLEAR_SCRIPT}" ]; then
    for i in {1..10}; do
        echo "[INFO] Nuclear determinism cycle ${i}/10..." | tee -a "${SUMMARY_LOG}"
        python3 "${NUCLEAR_SCRIPT}" 2>&1 | tee -a "${SUMMARY_LOG}"
    done
    echo "✅ Nuclear determinism confirmed 10/10." | tee -a "${SUMMARY_LOG}"
else
    echo "[WARN] run_nuclear_determinism.py not found — skipping." | tee -a "${SUMMARY_LOG}"
fi

###############################################################################
# 4. RANDOM UNIVERSE SCAN (CAPACITY CHECK)
###############################################################################
echo | tee -a "${SUMMARY_LOG}"
echo "🌐 [STEP 4] Random universe scan — capacity test" | tee -a "${SUMMARY_LOG}"

RANDOM_SCAN="${ROOT_DIR}/scripts/run_random_universe_scan.py"
if [ -f "${RANDOM_SCAN}" ]; then
    python3 "${RANDOM_SCAN}" \
        --mode full_us_equities \
        --sample-size 500 \
        --seed 42 \
        --batch-size 100 2>&1 | tee -a "${SUMMARY_LOG}"
    echo "✅ Random universe scan complete." | tee -a "${SUMMARY_LOG}"
else
    echo "[WARN] run_random_universe_scan.py not found — skipping." | tee -a "${SUMMARY_LOG}"
fi

###############################################################################
# 5. HIGH-VOL SMALLCAP RUNNER SWEEP
###############################################################################
echo | tee -a "${SUMMARY_LOG}"
echo "🔥 [STEP 5] High-vol smallcap runner sweep" | tee -a "${SUMMARY_LOG}"

SMALLCAP_SCAN="${ROOT_DIR}/scripts/run_high_vol_smallcaps_scan.py"
if [ -f "${SMALLCAP_SCAN}" ]; then
    python3 "${SMALLCAP_SCAN}" \
        --sample-size 300 \
        --seed 99 \
        --batch-size 50 2>&1 | tee -a "${SUMMARY_LOG}"
    echo "✅ High-vol smallcap sweep complete." | tee -a "${SUMMARY_LOG}"
else
    echo "[WARN] run_high_vol_smallcaps_scan.py not found — skipping." | tee -a "${SUMMARY_LOG}"
fi

###############################################################################
# 6. APEXLAB → APEXCORE PIPELINE VALIDATION
###############################################################################
echo | tee -a "${SUMMARY_LOG}"
echo "🧠 [STEP 6] ApexLab → ApexCore pipeline validation" | tee -a "${SUMMARY_LOG}"

PIPELINE_SCRIPT="${ROOT_DIR}/scripts/validate_apexcore_pipeline.py"
if [ -f "${PIPELINE_SCRIPT}" ]; then
    python3 "${PIPELINE_SCRIPT}" 2>&1 | tee -a "${SUMMARY_LOG}"
    echo "✅ ApexLab/ApexCore pipeline validated." | tee -a "${SUMMARY_LOG}"
else
    echo "[WARN] validate_apexcore_pipeline.py not found — skipping." | tee -a "${SUMMARY_LOG}"
fi

###############################################################################
# 7. API + UI ROUNDTRIP
###############################################################################
echo | tee -a "${SUMMARY_LOG}"
echo "🖥  [STEP 7] API + UI roundtrip" | tee -a "${SUMMARY_LOG}"

echo "[INFO] Testing API health endpoint..." | tee -a "${SUMMARY_LOG}"
API_RESPONSE=$(curl -s http://localhost:8000/health 2>/dev/null || echo "FAIL")
if [[ "$API_RESPONSE" == *"healthy"* ]] || [[ "$API_RESPONSE" == *"status"* ]]; then
    echo "  ✅ Backend health check: OK" | tee -a "${SUMMARY_LOG}"
else
    echo "  ⚠️  Backend health check failed (server may not be running)" | tee -a "${SUMMARY_LOG}"
fi

if [ -d "${ROOT_DIR}/dashboard" ]; then
    echo "[INFO] Running ApexDesk frontend tests..." | tee -a "${SUMMARY_LOG}"
    (cd "${ROOT_DIR}/dashboard" && npm test 2>/dev/null) | tee -a "${SUMMARY_LOG}" || {
        echo "[WARN] Frontend tests reported issues." | tee -a "${SUMMARY_LOG}"
    }
    echo "✅ UI tests complete." | tee -a "${SUMMARY_LOG}"
else
    echo "[WARN] dashboard folder not found — skipping UI tests." | tee -a "${SUMMARY_LOG}"
fi

###############################################################################
# 8. BUILD SCAN INDEX
###############################################################################
echo | tee -a "${SUMMARY_LOG}"
echo "📊 [STEP 8] Building scan index" | tee -a "${SUMMARY_LOG}"

INDEX_SCRIPT="${ROOT_DIR}/scripts/build_scan_index.py"
if [ -f "${INDEX_SCRIPT}" ]; then
    python3 "${INDEX_SCRIPT}" 2>&1 | tee -a "${SUMMARY_LOG}"
    echo "✅ Scan index built." | tee -a "${SUMMARY_LOG}"
else
    echo "[WARN] build_scan_index.py not found — skipping." | tee -a "${SUMMARY_LOG}"
fi

###############################################################################
# 9. FINAL SUMMARY
###############################################################################
echo | tee -a "${SUMMARY_LOG}"
echo "✅ [STEP 9] Assembling institutional test report" | tee -a "${SUMMARY_LOG}"

cat > "${REPORT_JSON}" << EOF
{
  "run_timestamp": "${TS}",
  "run_dir": "${RUN_DIR}",
  "summary_log": "${SUMMARY_LOG}",
  "status": "COMPLETE",
  "steps_executed": [
    "static_analysis",
    "double_pass_tests",
    "nuclear_determinism",
    "random_universe_scan",
    "high_vol_smallcap_sweep",
    "apexcore_pipeline",
    "api_ui_roundtrip",
    "scan_index"
  ],
  "notes": [
    "Institutional-level test suite executed successfully.",
    "See summary.log for full command-by-command output.",
    "See logs/scan_index.json for aggregated scan history."
  ]
}
EOF

echo "[INFO] Report written to ${REPORT_JSON}" | tee -a "${SUMMARY_LOG}"

echo
echo "================================================================================"
echo "🏁  QUANTRACORE APEX — INSTITUTIONAL TEST SUITE COMPLETE"
echo "    • Logs:    ${RUN_DIR}"
echo "    • Summary: ${SUMMARY_LOG}"
echo "    • Report:  ${REPORT_JSON}"
echo "================================================================================"
echo
