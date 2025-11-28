#!/usr/bin/env python3
"""
QuantraCore Apex — ApexDesk UI Deep Validation Harness
Author: Lamont Labs (Jesse J. Lamont)

Purpose
-------
Brutal validation script focused ONLY on the ApexDesk dashboard + UI layer.

Goals:
  • Prove that every available UI test, build, and lint pipeline passes.
  • Run UI tests multiple times to catch flaky behaviour.
  • Validate that the production build for the dashboard completes cleanly.
  • Capture dedicated logs + JSON summary for institutional-style audit.

What this script attempts to run (only if present in package.json / repo):
  1) npm run lint           — UI lint checks
  2) npm run typecheck      — TS type safety (if defined)
  3) npm test               — vitest/Jest component tests
  4) npm run test:ui        — dedicated UI test suite (if defined)
  5) npx playwright test    — Playwright e2e tests (if config present)
  6) npx cypress run        — Cypress e2e tests (if config present)
  7) npm run build          — production build (Vite/React/etc.)

Each command is run THREE times:
  • Round 1: CI profile (APEX_UI_ENV=ci)
  • Round 2: dashboard profile (APEX_UI_ENV=dashboard)
  • Round 3: stress profile (APEX_UI_ENV=stress)

Total: every UI command is executed 3× with different env flags.

Usage
-----
From repo root in Replit:

    python scripts/run_ui_deep_validation.py

Outputs
-------
Logs + JSON artifacts under:

    logs/ui_validation/<timestamp>/

  • ui_validation_log.txt          — human-readable log
  • ui_validation_summary.json     — aggregate results
  • ui_validation_runs.json        — per-run details (command, env, exit code, duration)

This script does NOT modify source code, only runs commands and records results.
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
OUTDIR_ROOT = ROOT / "logs" / "ui_validation"
OUTDIR_ROOT.mkdir(parents=True, exist_ok=True)

TS = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
OUTDIR = OUTDIR_ROOT / TS
OUTDIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTDIR / "ui_validation_log.txt"
SUMMARY_JSON = OUTDIR / "ui_validation_summary.json"
RUNS_JSON = OUTDIR / "ui_validation_runs.json"


@dataclass
class UIRunResult:
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
    run_id: int,
    extra_env: Optional[Dict[str, str]] = None,
) -> UIRunResult:
    """
    Execute a UI-related command with a given env profile.
    Captures stdout/stderr to a dedicated per-run log file.
    """
    log_path = OUTDIR / f"run_{run_id:03d}_{label.replace(' ', '_')}_env-{env_profile}_r{round_index}.log"

    env = os.environ.copy()
    env["APEX_UI_ENV"] = env_profile
    env["APEXDESK_UI_VALIDATION"] = "1"
    if extra_env:
        env.update(extra_env)

    log_line(
        f"[RUN {run_id:03d}] [{env_profile}] round={round_index} label={label}\n"
        f"          cmd={' '.join(command)}"
    )

    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as f:
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
        f"label={label} → exit={proc.returncode} duration={duration:.2f}s log={log_path}"
    )

    return UIRunResult(
        id=run_id,
        label=label,
        command=command,
        env_profile=env_profile,
        round_index=round_index,
        exit_code=proc.returncode,
        duration_sec=duration,
        log_path=str(log_path.relative_to(ROOT)),
    )


def read_package_json() -> Dict:
    pkg_file = ROOT / "package.json"
    if not pkg_file.exists():
        return {}
    try:
        return json.loads(pkg_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    log_line("=" * 79)
    log_line("QuantraCore Apex — ApexDesk UI Deep Validation Harness")
    log_line("=" * 79)
    log_line(f"Repo root        : {ROOT}")
    log_line(f"Output directory : {OUTDIR}")
    log_line(f"Timestamp (UTC)  : {TS}")
    log_line("")

    pkg = read_package_json()
    has_pkg = bool(pkg)
    if not has_pkg:
        log_line("[FATAL] package.json not found. UI validation cannot run.")
        return 1

    scripts = pkg.get("scripts", {}) or {}

    commands: List[Dict[str, List[str]]] = []

    if "lint" in scripts:
        commands.append({"label": "ui_lint", "cmd": ["npm", "run", "lint"]})

    if "typecheck" in scripts:
        commands.append({"label": "ui_typecheck", "cmd": ["npm", "run", "typecheck"]})

    if "test" in scripts:
        commands.append({"label": "ui_test_default", "cmd": ["npm", "test", "--", "--run"]})

    if "test:ui" in scripts:
        commands.append({"label": "ui_test_dedicated", "cmd": ["npm", "run", "test:ui"]})

    playwright_config = any(
        (ROOT / name).exists()
        for name in ["playwright.config.js", "playwright.config.ts"]
    )
    if playwright_config or (ROOT / "tests" / "e2e").exists():
        commands.append({"label": "ui_playwright_e2e", "cmd": ["npx", "playwright", "test"]})

    cypress_config = any(
        (ROOT / name).exists()
        for name in [
            "cypress.config.js",
            "cypress.config.ts",
            "cypress.json",
        ]
    )
    if cypress_config:
        commands.append({"label": "ui_cypress_e2e", "cmd": ["npx", "cypress", "run"]})

    if "build" in scripts:
        commands.append({"label": "ui_build_prod", "cmd": ["npm", "run", "build"]})

    if not commands:
        log_line("[FATAL] No UI-related commands detected in package.json or config files.")
        return 1

    log_line("Detected UI commands for validation:")
    for c in commands:
        log_line(f"  • {c['label']}: {' '.join(c['cmd'])}")
    log_line("")

    env_profiles = ["ci", "dashboard", "stress"]
    rounds_per_profile = 1

    run_results: List[UIRunResult] = []
    run_id = 0

    for env_profile in env_profiles:
        log_line("-" * 79)
        log_line(f"ENV PROFILE: {env_profile}")
        log_line("-" * 79)

        for round_idx in range(1, rounds_per_profile + 1):
            for c in commands:
                run_id += 1
                res = run_command(
                    label=c["label"],
                    command=c["cmd"],
                    env_profile=env_profile,
                    round_index=round_idx,
                    run_id=run_id,
                    extra_env={},
                )
                run_results.append(res)

                if res.exit_code != 0:
                    log_line("")
                    log_line("*" * 79)
                    log_line(f"[FAIL-FAST] UI command failed: label={res.label} "
                             f"env={res.env_profile} round={res.round_index}")
                    log_line(f"           See log: {res.log_path}")
                    log_line("*" * 79)
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
    log_line("=" * 79)
    log_line("UI DEEP VALIDATION SUMMARY")
    log_line("=" * 79)
    log_line(f"Total UI runs executed : {total_runs}")
    log_line(f"Passing runs           : {len(passes)}")
    log_line(f"Failing runs           : {len(failures)}")
    log_line("")

    if failures:
        log_line("Failing runs:")
        for r in failures:
            log_line(
                f"  • id={r.id:03d} label={r.label} env={r.env_profile} "
                f"round={r.round_index} exit={r.exit_code} log={r.log_path}"
            )
    else:
        log_line("All UI validation runs passed. Dashboard and UI stack are healthy.")
    log_line("=" * 79)

    RUNS_JSON.write_text(
        json.dumps([asdict(r) for r in run_results], indent=2),
        encoding="utf-8",
    )

    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "timestamp_utc": TS,
                "repo_root": str(ROOT),
                "output_dir": str(OUTDIR),
                "total_runs": total_runs,
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
