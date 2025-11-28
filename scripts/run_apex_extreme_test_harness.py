#!/usr/bin/env python3
"""
QuantraCore Apex — Extreme Test Harness (10× Amplified)
Author: Jesse J. Lamont — Lamont Labs

Purpose
-------
Central orchestrator that drives *every* important test dimension for QuantraCore Apex:

  • Core engine unit/integration/regression tests
  • Strategy & model robustness: backtests, walk-forward, Monte Carlo, stress scenarios
  • Risk controls & compliance-style tests
  • Data quality & historical coverage tests
  • Resilience & chaos / fault-injection tests
  • UI & dashboard: e2e, deep validation, button double-tests

It is designed to:
  • Run tests under THREE environments: ci, research, stress.
  • DOUBLE each scenario: two full passes per environment.
  • Capture all logs and a machine-readable summary for investor / institutional review.

This script does not change your codebase. It only executes commands and records results.

Usage
-----
From repo root:

    python scripts/run_apex_extreme_test_harness.py

Outputs
-------
All under: logs/apex_extreme_test/<timestamp>/

  • apex_extreme_log.txt         — human-readable overall log
  • apex_extreme_summary.json    — high-level counts / status
  • apex_extreme_runs.json       — one row per executed command
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
OUTDIR_ROOT = ROOT / "logs" / "apex_extreme_test"
OUTDIR_ROOT.mkdir(parents=True, exist_ok=True)

TS = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
OUTDIR = OUTDIR_ROOT / TS
OUTDIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTDIR / "apex_extreme_log.txt"
SUMMARY_JSON = OUTDIR / "apex_extreme_summary.json"
RUNS_JSON = OUTDIR / "apex_extreme_runs.json"
META_JSON = OUTDIR / "apex_extreme_meta.json"


@dataclass
class ExtremeRunResult:
    id: int
    group: str
    label: str
    command: List[str]
    env_profile: str
    round_index: int
    exit_code: int
    duration_sec: float
    log_path: str
    skipped: bool
    reason: str


def log(msg: str) -> None:
    msg = str(msg)
    print(msg)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def write_meta() -> None:
    meta = {
        "timestamp_utc": TS,
        "repo_root": str(ROOT),
        "outdir": str(OUTDIR.relative_to(ROOT)),
        "env_profiles": ["ci", "research", "stress"],
        "rounds_per_profile": 2,
    }
    META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def command_exists(cmd: List[str]) -> bool:
    """
    Basic existence check: for Python scripts, ensure file exists;
    for shell scripts, ensure file exists; for npm-related commands, rely on package.json.
    """
    if not cmd:
        return False
    if cmd[0] == "python" and len(cmd) >= 2 and cmd[1].endswith(".py"):
        path = ROOT / cmd[1]
        return path.exists()
    if cmd[0] in ("bash", "sh") and len(cmd) >= 2:
        path = ROOT / cmd[1]
        return path.exists()
    if cmd[0] in ("npm", "npx"):
        return (ROOT / "package.json").exists()
    return True


def run_command(
    group: str,
    label: str,
    command: List[str],
    env_profile: str,
    round_index: int,
    run_id: int,
    extra_env: Optional[Dict[str, str]] = None,
) -> ExtremeRunResult:
    """
    Execute a command with a given env profile, capturing output to a log file.
    """
    run_log = OUTDIR / f"run_{run_id:03d}_{group}_{label}_env-{env_profile}_r{round_index}.log"

    env = os.environ.copy()
    env["APEX_ENV"] = env_profile
    env["APEX_EXTREME_TEST"] = "1"
    if extra_env:
        env.update(extra_env)

    log(
        f"[RUN {run_id:03d}] [{group}] [{env_profile}] round={round_index} label={label}\n"
        f"          cmd={' '.join(command)}"
    )

    start = time.perf_counter()
    with run_log.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        proc.wait()
    duration = time.perf_counter() - start

    log(
        f"[RUN {run_id:03d}] [{group}] [{env_profile}] round={round_index} "
        f"label={label} → exit={proc.returncode} duration={duration:.2f}s log={run_log}"
    )

    return ExtremeRunResult(
        id=run_id,
        group=group,
        label=label,
        command=command,
        env_profile=env_profile,
        round_index=round_index,
        exit_code=proc.returncode,
        duration_sec=duration,
        log_path=str(run_log.relative_to(ROOT)),
        skipped=False,
        reason="",
    )


def build_test_catalog() -> List[Dict[str, any]]:
    """
    Define all extreme tests we want to orchestrate.
    """
    catalog: List[Dict[str, any]] = []

    if (ROOT / "tests").exists():
        catalog.append({
            "group": "core_tests",
            "label": "pytest_full",
            "cmd": ["python", "-m", "pytest", "-q", "tests/"],
        })

    if (ROOT / "scripts" / "run_protocol_triple_validation.py").exists():
        catalog.append({
            "group": "core_tests",
            "label": "protocol_triple_validation",
            "cmd": ["python", "scripts/run_protocol_triple_validation.py"],
        })

    if (ROOT / "scripts" / "run_backtest_matrix.py").exists():
        catalog.append({
            "group": "strategy_and_model",
            "label": "backtest_matrix",
            "cmd": ["python", "scripts/run_backtest_matrix.py"],
        })

    if (ROOT / "scripts" / "run_walkforward_matrix.py").exists():
        catalog.append({
            "group": "strategy_and_model",
            "label": "walkforward_matrix",
            "cmd": ["python", "scripts/run_walkforward_matrix.py"],
        })

    if (ROOT / "scripts" / "run_monte_carlo_equity_sim.py").exists():
        catalog.append({
            "group": "strategy_and_model",
            "label": "monte_carlo_equity",
            "cmd": ["python", "scripts/run_monte_carlo_equity_sim.py"],
        })

    if (ROOT / "scripts" / "run_apexcore_alignment_tests.py").exists():
        catalog.append({
            "group": "strategy_and_model",
            "label": "apexcore_alignment",
            "cmd": ["python", "scripts/run_apexcore_alignment_tests.py"],
        })

    if (ROOT / "scripts" / "run_risk_controls_validation.py").exists():
        catalog.append({
            "group": "risk_and_controls",
            "label": "risk_controls_validation",
            "cmd": ["python", "scripts/run_risk_controls_validation.py"],
        })

    if (ROOT / "scripts" / "zde_validation.py").exists():
        catalog.append({
            "group": "risk_and_controls",
            "label": "zde_validation",
            "cmd": ["python", "scripts/zde_validation.py"],
        })

    if (ROOT / "scripts" / "run_kill_switch_tests.py").exists():
        catalog.append({
            "group": "risk_and_controls",
            "label": "kill_switch_tests",
            "cmd": ["python", "scripts/run_kill_switch_tests.py"],
        })

    if (ROOT / "scripts" / "run_data_quality_checks.py").exists():
        catalog.append({
            "group": "data_and_quality",
            "label": "data_quality_checks",
            "cmd": ["python", "scripts/run_data_quality_checks.py"],
        })

    if (ROOT / "scripts" / "run_random_universe_scan.py").exists():
        catalog.append({
            "group": "data_and_quality",
            "label": "random_universe_scan",
            "cmd": ["python", "scripts/run_random_universe_scan.py", "--mode", "full_us_equities"],
        })

    if (ROOT / "scripts" / "run_high_vol_smallcaps_scan.py").exists():
        catalog.append({
            "group": "data_and_quality",
            "label": "high_vol_smallcaps_scan",
            "cmd": ["python", "scripts/run_high_vol_smallcaps_scan.py"],
        })

    if (ROOT / "scripts" / "run_latency_stress_test.py").exists():
        catalog.append({
            "group": "resilience_and_chaos",
            "label": "latency_stress",
            "cmd": ["python", "scripts/run_latency_stress_test.py"],
        })

    if (ROOT / "scripts" / "run_soak_test.py").exists():
        catalog.append({
            "group": "resilience_and_chaos",
            "label": "soak_test",
            "cmd": ["python", "scripts/run_soak_test.py"],
        })

    if (ROOT / "scripts" / "run_chaos_fault_injection.py").exists():
        catalog.append({
            "group": "resilience_and_chaos",
            "label": "chaos_fault_injection",
            "cmd": ["python", "scripts/run_chaos_fault_injection.py"],
        })

    if (ROOT / "scripts" / "run_ui_deep_validation.py").exists():
        catalog.append({
            "group": "ui_and_ux",
            "label": "ui_deep_validation",
            "cmd": ["python", "scripts/run_ui_deep_validation.py"],
        })

    if (ROOT / "package.json").exists():
        catalog.append({
            "group": "ui_and_ux",
            "label": "frontend_tests",
            "cmd": ["npm", "test", "--", "--run"],
        })

    return catalog


def main() -> int:
    write_meta()

    log("=" * 80)
    log("QuantraCore Apex — Extreme Test Harness (10× Amplified)")
    log("=" * 80)
    log(f"Repo root   : {ROOT}")
    log(f"Output dir  : {OUTDIR}")
    log(f"Timestamp   : {TS} (UTC)")
    log("")

    test_catalog = build_test_catalog()
    if not test_catalog:
        log("[FATAL] No test commands discovered. Add scripts/tests and re-run.")
        return 1

    log("Planned extreme test commands:")
    for entry in test_catalog:
        log(f"  • [{entry['group']}] {entry['label']}: {' '.join(entry['cmd'])}")
    log("")

    env_profiles = ["ci", "research", "stress"]
    rounds_per_profile = 2

    run_results: List[ExtremeRunResult] = []
    run_id = 0

    for env_profile in env_profiles:
        log("-" * 80)
        log(f"ENV PROFILE: {env_profile}")
        log("-" * 80)

        for round_idx in range(1, rounds_per_profile + 1):
            log(f"--------------------------- ROUND {round_idx}/{rounds_per_profile} ({env_profile}) ---------------------------")

            for entry in test_catalog:
                group = entry["group"]
                label = entry["label"]
                cmd = entry["cmd"]

                run_id += 1

                if not command_exists(cmd):
                    reason = "command_or_script_missing"
                    log(f"[SKIP {run_id:03d}] [{group}] [{env_profile}] label={label} → {reason}")
                    run_results.append(
                        ExtremeRunResult(
                            id=run_id,
                            group=group,
                            label=label,
                            command=cmd,
                            env_profile=env_profile,
                            round_index=round_idx,
                            exit_code=0,
                            duration_sec=0.0,
                            log_path="",
                            skipped=True,
                            reason=reason,
                        )
                    )
                    continue

                res = run_command(
                    group=group,
                    label=label,
                    command=cmd,
                    env_profile=env_profile,
                    round_index=round_idx,
                    run_id=run_id,
                    extra_env={},
                )
                run_results.append(res)

    total_runs = len(run_results)
    executed = [r for r in run_results if not r.skipped]
    skipped = [r for r in run_results if r.skipped]
    failures = [r for r in executed if r.exit_code != 0]
    passes = [r for r in executed if r.exit_code == 0]

    log("")
    log("=" * 80)
    log("EXTREME TEST SUMMARY")
    log("=" * 80)
    log(f"Total planned runs     : {total_runs}")
    log(f"Executed (not skipped) : {len(executed)}")
    log(f"Skipped (not available): {len(skipped)}")
    log(f"Passes                 : {len(passes)}")
    log(f"Failures               : {len(failures)}")
    log("")

    if failures:
        log("Failing runs:")
        for r in failures:
            log(
                f"  • id={r.id:03d} [{r.group}] {r.label} "
                f"env={r.env_profile} round={r.round_index} "
                f"exit={r.exit_code} log={r.log_path}"
            )
    else:
        log("All executed extreme tests passed.")
    log("=" * 80)
    log(f"Summary JSON : {SUMMARY_JSON.relative_to(ROOT)}")
    log(f"Runs JSON    : {RUNS_JSON.relative_to(ROOT)}")
    log(f"Meta JSON    : {META_JSON.relative_to(ROOT)}")
    log("=" * 80)

    RUNS_JSON.write_text(
        json.dumps([asdict(r) for r in run_results], indent=2),
        encoding="utf-8",
    )

    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "timestamp_utc": TS,
                "repo_root": str(ROOT),
                "output_dir": str(OUTDIR.relative_to(ROOT)),
                "total_runs": total_runs,
                "executed": len(executed),
                "skipped": len(skipped),
                "passes": len(passes),
                "failures": len(failures),
                "failed_run_ids": [r.id for r in failures],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
