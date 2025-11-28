#!/usr/bin/env python3
"""
QuantraCore Apex — Institutional Battle-Test Harness
Author: Lamont Labs (Jesse J. Lamont)

Purpose
-------
Brutal, institutional-grade battle test for the full QuantraCore Apex stack.

Goals:
  • Exercise every major test + verification path MULTIPLE times.
  • Run under two environments (CI-style + research-style) to mimic real usage.
  • Double every scenario: each command is executed at least twice per environment.
  • Capture full logs + exit codes for post-mortem / audit.
  • Fail-fast on any error, with a clear summary at the end.

What this script runs (when available in the repo):
  1) Backend test suite via pytest                      (core engine + protocols + scanner)
  2) Focused scanner tests (if tests/scanner* exists)
  3) Frontend test suite via npm test (if package.json exists)
  4) Protocol triple-run validation (if scripts/run_protocol_triple_validation.py exists)
  5) Zero-Doubt Verification script (if scripts/zero_doubt_verification.sh exists)

All of the above are run:
  • Under APEX_ENV=ci        (CI / institutional mode)
  • Under APEX_ENV=research  (research / exploratory mode)

Each command is executed twice per environment (so "doubled"),
for a total of 4 runs per command (2 envs × 2 passes).

Run
---
From repo root:

    python scripts/run_institutional_battle_test.py

Outputs
-------
Logs and machine-readable summary under:

    logs/battle_test/<timestamp>/

  • battle_test_log.txt          — human-readable log
  • battle_test_summary.json     — aggregate results
  • battle_test_runs.json        — detailed per-run info (command, env, exit code, duration)
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
OUTDIR_ROOT = ROOT / "logs" / "battle_test"
OUTDIR_ROOT.mkdir(parents=True, exist_ok=True)

TS = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
OUTDIR = OUTDIR_ROOT / TS
OUTDIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTDIR / "battle_test_log.txt"
SUMMARY_JSON = OUTDIR / "battle_test_summary.json"
RUNS_JSON = OUTDIR / "battle_test_runs.json"


@dataclass
class BattleRunResult:
    id: int
    label: str
    command: List[str]
    env_profile: str
    round_index: int
    exit_code: int
    duration_sec: float
    log_path: str


def log_line(msg: str) -> None:
    msg = str(msg)
    print(msg)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run_command(
    label: str,
    command: List[str],
    env_profile: str,
    round_index: int,
    extra_env: Optional[Dict[str, str]] = None,
    run_id: int = 0,
) -> BattleRunResult:
    """
    Execute command with specific env profile, capture duration and exit code.
    Writes per-run stdout/stderr into a dedicated log file under OUTDIR.
    """
    run_log = OUTDIR / f"run_{run_id:03d}_{label.replace(' ', '_')}_env-{env_profile}_r{round_index}.log"

    env = os.environ.copy()
    env["APEX_ENV"] = env_profile
    env["APEX_BATTLE_TEST"] = "1"
    if extra_env:
        env.update(extra_env)

    log_line(
        f"[RUN {run_id:03d}] [{env_profile}] round={round_index} label={label}\n"
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

    log_line(
        f"[RUN {run_id:03d}] [{env_profile}] round={round_index} "
        f"label={label} → exit={proc.returncode}  duration={duration:.2f}s  log={run_log}"
    )

    return BattleRunResult(
        id=run_id,
        label=label,
        command=command,
        env_profile=env_profile,
        round_index=round_index,
        exit_code=proc.returncode,
        duration_sec=duration,
        log_path=str(run_log.relative_to(ROOT)),
    )


def main() -> int:
    log_line("=" * 80)
    log_line("QuantraCore Apex — Institutional Battle Test")
    log_line("=" * 80)
    log_line(f"Repo root     : {ROOT}")
    log_line(f"Output folder : {OUTDIR}")
    log_line(f"Timestamp UTC : {TS}")
    log_line("")

    if not (ROOT / "pyproject.toml").exists() and not (ROOT / "requirements.txt").exists():
        log_line("[WARN] No pyproject.toml or requirements.txt detected. "
                 "Assuming Python environment is already prepared.")

    has_tests_dir = (ROOT / "tests").exists()
    has_scanner_tests = (ROOT / "tests" / "scanner").exists() or any(
        p.name.startswith("test_scanner") for p in (ROOT / "tests").glob("test_*")
    ) if has_tests_dir else False

    has_pkg_json = (ROOT / "package.json").exists()
    has_zero_doubt = (ROOT / "scripts" / "zero_doubt_verification.sh").exists()
    has_protocol_triple = (ROOT / "scripts" / "run_protocol_triple_validation.py").exists()

    log_line("Detected components:")
    log_line(f"  • tests/ directory          : {'YES' if has_tests_dir else 'NO'}")
    log_line(f"  • scanner-focused tests     : {'YES' if has_scanner_tests else 'NO'}")
    log_line(f"  • package.json (frontend)   : {'YES' if has_pkg_json else 'NO'}")
    log_line(f"  • zero_doubt_verification   : {'YES' if has_zero_doubt else 'NO'}")
    log_line(f"  • protocol triple validation: {'YES' if has_protocol_triple else 'NO'}")
    log_line("")

    commands: List[Dict[str, any]] = []

    if has_tests_dir:
        commands.append(
            {
                "label": "backend_pytest_full",
                "cmd": ["python", "-m", "pytest", "-q", "tests/"],
            }
        )

    if has_scanner_tests:
        scanner_cmd = (
            ["python", "-m", "pytest", "-q", "tests/scanner"]
            if (ROOT / "tests" / "scanner").exists()
            else ["python", "-m", "pytest", "-q", "-k", "scanner"]
        )
        commands.append(
            {
                "label": "scanner_focus_tests",
                "cmd": scanner_cmd,
            }
        )

    if has_pkg_json:
        commands.append(
            {
                "label": "frontend_npm_test",
                "cmd": ["npm", "test", "--", "--run"],
            }
        )

    if has_protocol_triple:
        commands.append(
            {
                "label": "protocol_triple_validation",
                "cmd": ["python", "scripts/run_protocol_triple_validation.py"],
            }
        )

    if has_zero_doubt:
        commands.append(
            {
                "label": "zero_doubt_verification",
                "cmd": ["bash", "scripts/zero_doubt_verification.sh"],
            }
        )

    if not commands:
        log_line("[FATAL] No suitable commands discovered for battle testing. "
                 "Ensure tests + scripts are present.")
        return 1

    log_line("Battle test command plan:")
    for c in commands:
        log_line(f"  • {c['label']}: {' '.join(c['cmd'])}")
    log_line("")

    env_profiles = ["ci", "research"]
    rounds_per_profile = 2

    run_results: List[BattleRunResult] = []
    run_id = 0

    for env_profile in env_profiles:
        log_line("=" * 80)
        log_line(f"ENV PROFILE: {env_profile}")
        log_line("=" * 80)
        for round_idx in range(1, rounds_per_profile + 1):
            log_line(f"-------------------------- ROUND {round_idx}/{rounds_per_profile} "
                     f"({env_profile}) --------------------------")

            for c in commands:
                run_id += 1
                try:
                    res = run_command(
                        label=c["label"],
                        command=c["cmd"],
                        env_profile=env_profile,
                        round_index=round_idx,
                        extra_env=None,
                        run_id=run_id,
                    )
                    run_results.append(res)

                    if res.exit_code != 0:
                        log_line("")
                        log_line("*" * 80)
                        log_line(f"[FAIL-FAST] Command failed: label={res.label}, "
                                 f"env={res.env_profile}, round={res.round_index}")
                        log_line(f"          See log: {res.log_path}")
                        log_line("*" * 80)
                        break
                except FileNotFoundError:
                    log_line(f"[ERROR] Command not found for label={c['label']} "
                             f"(cmd={' '.join(c['cmd'])})")
                    run_results.append(
                        BattleRunResult(
                            id=run_id,
                            label=c["label"],
                            command=c["cmd"],
                            env_profile=env_profile,
                            round_index=round_idx,
                            exit_code=127,
                            duration_sec=0.0,
                            log_path="",
                        )
                    )
                    break
            else:
                continue
            break
        else:
            continue
        break

    total_runs = len(run_results)
    failures = [r for r in run_results if r.exit_code != 0]
    passes = [r for r in run_results if r.exit_code == 0]

    log_line("")
    log_line("=" * 80)
    log_line("BATTLE TEST SUMMARY")
    log_line("=" * 80)
    log_line(f"Total runs executed : {total_runs}")
    log_line(f"Passed              : {len(passes)}")
    log_line(f"Failed              : {len(failures)}")
    log_line("")

    if failures:
        log_line("Failed runs:")
        for r in failures:
            log_line(
                f"  • id={r.id:03d} label={r.label} env={r.env_profile} "
                f"round={r.round_index} exit={r.exit_code} log={r.log_path}"
            )
    else:
        log_line("All battle-test runs passed. System survived institutional-level barrage.")
    log_line("=" * 80)

    runs_payload = [asdict(r) for r in run_results]
    RUNS_JSON.write_text(json.dumps(runs_payload, indent=2), encoding="utf-8")

    summary_payload = {
        "timestamp_utc": TS,
        "repo_root": str(ROOT),
        "output_dir": str(OUTDIR),
        "total_runs": total_runs,
        "passes": len(passes),
        "failures": len(failures),
        "failed_run_ids": [r.id for r in failures],
    }
    SUMMARY_JSON.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
