#!/usr/bin/env python3
"""
QuantraCore Apex™ — Institutional Superrunner (Replit Edition)

Purpose
-------
One-button, institutional-grade, "kick ass" runner for the entire Apex ecosystem.

This script is designed for Replit and institutional environments to:

  1. Bootstrap hardware-aware scaling (apex_institutional_bootstrap).
  2. Run all existing heavy-duty test batteries:
       - Extreme Test
       - Nuclear Deep-Dive
       - Full pytest suite
  3. Run the full Protocol Scenario Matrix (if available).
  4. Run performance/latency/throughput tests (if present).
  5. Optionally run long-duration stability tests (configurable).
  6. Emit a single, professional summary (JSON + TXT) for auditors/investors.

It does NOT modify system-level configs. It only runs your repo scripts/tests
and writes reports/logs under:

  - reports/institutional_superrun/
  - logs/institutional_superrun/

Usage (from Replit shell or Agent)
----------------------------------
  $ python scripts/apex_institutional_superrun.py

Optional flags:
  --skip-bootstrap       : do not run apex_institutional_bootstrap
  --skip-extreme         : skip extreme test suite
  --skip-nuclear         : skip nuclear deep-dive suite
  --skip-matrix          : skip protocol scenario matrix
  --skip-perf            : skip perf/latency/throughput tests
  --skip-longrun         : skip long-duration stability test
  --longrun-minutes N    : override default long-run duration (default: 60)
  --dry-run              : print what would be run, but do not execute anything
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class StepResult:
    name: str
    description: str
    command: List[str]
    started_utc: str
    finished_utc: str
    return_code: int
    log_path: str


@dataclass
class SuperrunSummary:
    system: str
    version: str
    generated_utc: str
    host: str
    steps: List[StepResult]
    overall_status: str


def _utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_dirs() -> Dict[str, Path]:
    reports_root = Path("reports") / "institutional_superrun"
    logs_root = Path("logs") / "institutional_superrun"
    reports_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    return {"reports_root": reports_root, "logs_root": logs_root}


def _run_step(
    name: str,
    description: str,
    cmd: List[str],
    logs_root: Path,
    env: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
) -> StepResult:
    started = _utc_now_iso()
    log_file = logs_root / f"{name}_{_dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"

    if dry_run:
        body = [
            f"[DRY-RUN] {name}",
            f"Description : {description}",
            f"Command     : {' '.join(cmd)}",
        ]
        log_file.write_text("\n".join(body), encoding="utf-8")
        finished = _utc_now_iso()
        return StepResult(
            name=name,
            description=description,
            command=cmd,
            started_utc=started,
            finished_utc=finished,
            return_code=0,
            log_path=str(log_file),
        )

    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    proc = subprocess.run(cmd, text=True, capture_output=True, env=full_env)

    body = [
        f"Step: {name}",
        f"Description : {description}",
        f"Command     : {' '.join(cmd)}",
        f"Started UTC : {started}",
        f"Finished UTC: {_utc_now_iso()}",
        f"Return code : {proc.returncode}",
        "",
        "----- STDOUT -----",
        proc.stdout,
        "----- STDERR -----",
        proc.stderr,
    ]
    log_file.write_text("\n".join(body), encoding="utf-8")

    finished = _utc_now_iso()
    return StepResult(
        name=name,
        description=description,
        command=cmd,
        started_utc=started,
        finished_utc=finished,
        return_code=proc.returncode,
        log_path=str(log_file),
    )


def _file_exists(path: str) -> bool:
    return Path(path).is_file()


def _dir_exists(path: str) -> bool:
    return Path(path).is_dir()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuantraCore Apex — Institutional Superrunner")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Skip hardware bootstrap")
    parser.add_argument("--skip-extreme", action="store_true", help="Skip extreme test suite")
    parser.add_argument("--skip-nuclear", action="store_true", help="Skip nuclear deep-dive suite")
    parser.add_argument("--skip-matrix", action="store_true", help="Skip protocol scenario matrix")
    parser.add_argument("--skip-perf", action="store_true", help="Skip perf/latency tests")
    parser.add_argument("--skip-longrun", action="store_true", help="Skip long-duration stability")
    parser.add_argument(
        "--longrun-minutes",
        type=int,
        default=60,
        help="Long-run stability duration in minutes (default: 60)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run, but do not execute any commands.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dirs = _ensure_dirs()

    host = os.getenv("HOSTNAME") or os.getenv("COMPUTERNAME") or "unknown-host"
    steps: List[StepResult] = []

    print("=" * 60)
    print("QuantraCore Apex — Institutional Superrunner")
    print("=" * 60)
    print(f"Host       : {host}")
    print(f"Dry run    : {args.dry_run}")
    print(f"Logs dir   : {dirs['logs_root']}")
    print(f"Reports dir: {dirs['reports_root']}")
    print()

    # 1. Hardware-aware bootstrap
    if not args.skip_bootstrap:
        if _file_exists("scripts/apex_institutional_bootstrap.py"):
            cmd = [sys.executable, "scripts/apex_institutional_bootstrap.py"]
            print("[RUN] bootstrap...")
            steps.append(
                _run_step(
                    name="bootstrap",
                    description="Hardware profiling + scaling profile (apex_institutional_bootstrap).",
                    cmd=cmd,
                    logs_root=dirs["logs_root"],
                    dry_run=args.dry_run,
                )
            )
        else:
            print("[WARN] scripts/apex_institutional_bootstrap.py not found; skipping bootstrap.")
    else:
        print("[SKIP] bootstrap")

    # 2. Extreme Test suite
    if not args.skip_extreme:
        description = "Full Apex Extreme Test suite (CI+research+stress)."
        if _file_exists("scripts/run_apex_extreme_test_harness.py"):
            cmd = [sys.executable, "scripts/run_apex_extreme_test_harness.py"]
        elif _dir_exists("tests"):
            cmd = ["pytest", "tests", "-m", "apex_extreme", "--disable-warnings", "--maxfail=1"]
        else:
            cmd = ["echo", "No tests directory; extreme test skipped logically."]
            description += " (no tests/ dir detected; echo-only fallback)"

        print("[RUN] extreme_test...")
        steps.append(
            _run_step(
                name="extreme_test",
                description=description,
                cmd=cmd,
                logs_root=dirs["logs_root"],
                dry_run=args.dry_run,
            )
        )
    else:
        print("[SKIP] extreme tests")

    # 3. Nuclear Deep-Dive suite
    if not args.skip_nuclear:
        description = "Apex Nuclear Deep-Dive (multi-cycle determinism validation)."
        if _file_exists("scripts/run_nuclear_deep_dive.py"):
            cmd = [sys.executable, "scripts/run_nuclear_deep_dive.py"]
        elif _dir_exists("tests"):
            cmd = ["pytest", "tests", "-m", "apex_nuclear", "--disable-warnings", "--maxfail=1"]
        else:
            cmd = ["echo", "No tests directory; nuclear test skipped logically."]
            description += " (no tests/ dir detected; echo-only fallback)"

        print("[RUN] nuclear_test...")
        steps.append(
            _run_step(
                name="nuclear_test",
                description=description,
                cmd=cmd,
                logs_root=dirs["logs_root"],
                dry_run=args.dry_run,
            )
        )
    else:
        print("[SKIP] nuclear tests")

    # 4. Full pytest test suite (baseline)
    if _dir_exists("tests"):
        cmd = ["pytest", "tests", "--disable-warnings", "--maxfail=1"]
        print("[RUN] pytest_full...")
        steps.append(
            _run_step(
                name="pytest_full",
                description="Full pytest suite over tests/ (unit + integration).",
                cmd=cmd,
                logs_root=dirs["logs_root"],
                dry_run=args.dry_run,
            )
        )
    else:
        print("[WARN] tests/ directory not found; full pytest suite skipped.")

    # 5. Protocol Scenario Matrix runner (if present)
    if not args.skip_matrix:
        if _file_exists("scripts/run_apex_protocol_matrix.py"):
            cmd = [sys.executable, "scripts/run_apex_protocol_matrix.py"]
            print("[RUN] protocol_matrix...")
            steps.append(
                _run_step(
                    name="protocol_matrix",
                    description="Protocol Scenario Matrix across all protocols × scenarios.",
                    cmd=cmd,
                    logs_root=dirs["logs_root"],
                    dry_run=args.dry_run,
                )
            )
        else:
            print("[WARN] scripts/run_apex_protocol_matrix.py not found; skipping matrix.")
    else:
        print("[SKIP] protocol scenario matrix")

    # 6. Performance / latency / throughput tests (optional)
    if not args.skip_perf:
        if _dir_exists("tests/perf"):
            cmd = ["pytest", "tests/perf", "--disable-warnings", "--maxfail=1"]
            print("[RUN] perf_tests...")
            steps.append(
                _run_step(
                    name="perf_tests",
                    description="Performance/latency/throughput tests under tests/perf/ (if defined).",
                    cmd=cmd,
                    logs_root=dirs["logs_root"],
                    dry_run=args.dry_run,
                )
            )
        else:
            print("[INFO] tests/perf/ not found; performance tests skipped.")
    else:
        print("[SKIP] perf tests")

    # 7. Long-duration stability test (optional)
    if not args.skip_longrun:
        if _dir_exists("tests/stability"):
            env = {"APEX_LONGRUN_MINUTES": str(args.longrun_minutes)}
            cmd = ["pytest", "tests/stability", "--disable-warnings", "--maxfail=1"]
            print(f"[RUN] longrun_stability ({args.longrun_minutes} min)...")
            steps.append(
                _run_step(
                    name="longrun_stability",
                    description=(
                        f"Long-duration stability test, target duration {args.longrun_minutes} min."
                    ),
                    cmd=cmd,
                    logs_root=dirs["logs_root"],
                    env=env,
                    dry_run=args.dry_run,
                )
            )
        else:
            print("[INFO] tests/stability/ not found; longrun stability tests skipped.")
    else:
        print("[SKIP] long-duration stability")

    # Summary
    overall_status = "OK"
    for s in steps:
        if s.return_code != 0:
            overall_status = "FAILED"
            break

    summary = SuperrunSummary(
        system="QuantraCore Apex",
        version="v9.0-A (Institutional Hardening)",
        generated_utc=_utc_now_iso(),
        host=host,
        steps=steps,
        overall_status=overall_status,
    )

    # Write JSON summary
    summary_json = dirs["reports_root"] / "apex_institutional_superrun_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "system": summary.system,
                "version": summary.version,
                "generated_utc": summary.generated_utc,
                "host": summary.host,
                "overall_status": summary.overall_status,
                "steps": [asdict(s) for s in summary.steps],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Write human-readable TXT summary
    summary_txt = dirs["reports_root"] / "apex_institutional_superrun_summary.txt"
    lines: List[str] = [
        "QuantraCore Apex — Institutional Superrun Summary",
        "=" * 50,
        f"Generated UTC : {summary.generated_utc}",
        f"Host          : {summary.host}",
        f"System        : {summary.system}",
        f"Version       : {summary.version}",
        f"Overall       : {summary.overall_status}",
        "",
        "Steps:",
    ]
    for s in summary.steps:
        lines.extend([
            f"- {s.name}",
            f"  Description : {s.description}",
            f"  Started UTC : {s.started_utc}",
            f"  Finished UTC: {s.finished_utc}",
            f"  Return code : {s.return_code}",
            f"  Log         : {s.log_path}",
            "",
        ])
    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("-" * 60)
    print("Institutional superrun completed.")
    print(f"Overall status : {overall_status}")
    print(f"Summary JSON   : {summary_json}")
    print(f"Summary TXT    : {summary_txt}")


if __name__ == "__main__":
    main()
