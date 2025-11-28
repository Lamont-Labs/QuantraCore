#!/usr/bin/env python3
"""
QuantraCore Apex™ — Full Protocol Scenario Matrix Runner (Replit Edition)

Purpose
-------
This script is designed to be run by Replit (Agent or normal shell) to:

  1. Exercise *every* QuantraCore Apex protocol (Tier, Learning, MonsterRunner, Omega)
     across a broad grid of market scenarios.
  2. Persist all results into the repo in a professional, machine-readable layout.
  3. Explicitly label all results as **free-tier data only** so institutional re-runs
     can be added later without overwriting provenance.

How it works
------------
- For each protocol_id and scenario_id, the script:
    - Sets environment variables describing the protocol + scenario:
        APEX_PROTOCOL_ID, APEX_SCENARIO_ID, APEX_SCENARIO_JSON
    - Invokes `pytest` over your test suite.
      Your existing tests can optionally read those env vars to specialize behaviour.
    - Captures:
        - JUnit XML (raw test data)
        - stdout/stderr log
        - a high-level JSON summary
    - Marks every record with `"data_tier": "free"`.

- At the end, it writes a master manifest that Replit, dashboards, and external
  reviewers can use.

Assumptions
-----------
- You already have a pytest-based test suite wired to QuantraCore Apex.
- Replit can run: `python scripts/run_apex_protocol_matrix.py`.
- If your tests choose not to read the APEX_* env vars yet, this script still runs
  and records results; you can wire in scenario-aware behaviour later without
  changing this orchestrator.

Directory layout created
------------------------
reports/
  protocol_matrix/
    raw_junit/
      <run_id>.xml
    summaries/
      <run_id>.json
    manifest.json

logs/
  protocol_matrix/
    <run_id>.log

You can safely commit these directories to the repo.
"""

from __future__ import annotations

import datetime as _dt
import itertools
import json
import os
import subprocess
import textwrap
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List


TIER_PROTOCOLS: List[str] = [f"T{str(i).zfill(2)}" for i in range(1, 81)]
LEARNING_PROTOCOLS: List[str] = [f"LP{str(i).zfill(2)}" for i in range(1, 26)]
MONSTER_PROTOCOLS: List[str] = [f"MR{str(i).zfill(2)}" for i in range(1, 6)]
OMEGA_PROTOCOLS: List[str] = [f"Ω{str(i)}" for i in range(1, 6)]

ALL_PROTOCOLS: List[str] = (
    TIER_PROTOCOLS + LEARNING_PROTOCOLS + MONSTER_PROTOCOLS + OMEGA_PROTOCOLS
)

SCENARIOS: List[Dict[str, str]] = [
    {
        "id": "bull_trend_daily_normal_liq_megacap",
        "market_regime": "bull_trend",
        "timeframe": "swing_daily",
        "volatility_regime": "vol_normal",
        "liquidity": "liquidity_ultra_high",
        "cap_tier": "mega_cap",
        "gap_struct": "no_gap_normal_session",
        "trend_stage": "early_trend_launch",
        "runner_context": "standard_runner",
        "zde_context": "zde_tolerance_band",
        "execution_env": "ci_batch",
    },
    {
        "id": "bear_trend_daily_high_liq_midcap",
        "market_regime": "bear_trend",
        "timeframe": "swing_daily",
        "volatility_regime": "vol_high",
        "liquidity": "liquidity_high",
        "cap_tier": "mid_cap",
        "gap_struct": "no_gap_normal_session",
        "trend_stage": "mature_trend",
        "runner_context": "no_runner_normal",
        "zde_context": "zde_fail",
        "execution_env": "research_batch",
    },
    {
        "id": "crash_intraday5_extreme_smallcap",
        "market_regime": "crash_event",
        "timeframe": "intraday_5m",
        "volatility_regime": "vol_extreme",
        "liquidity": "liquidity_mid",
        "cap_tier": "small_cap",
        "gap_struct": "gap_down_open",
        "trend_stage": "late_exhaustion",
        "runner_context": "failed_runner_attempt",
        "zde_context": "zde_fail",
        "execution_env": "stress_batch",
    },
    {
        "id": "parabolic_intraday1_extreme_microcap_monster",
        "market_regime": "parabolic_up",
        "timeframe": "intraday_1m",
        "volatility_regime": "vol_extreme",
        "liquidity": "liquidity_low",
        "cap_tier": "micro_cap",
        "gap_struct": "gap_up_open",
        "trend_stage": "early_trend_launch",
        "runner_context": "monster_runner",
        "zde_context": "zde_true",
        "execution_env": "nuclear_multi_cycle",
    },
    {
        "id": "sideways_daily_low_megacap",
        "market_regime": "sideways_range",
        "timeframe": "swing_daily",
        "volatility_regime": "vol_very_low",
        "liquidity": "liquidity_ultra_high",
        "cap_tier": "mega_cap",
        "gap_struct": "no_gap_normal_session",
        "trend_stage": "consolidation_before_break",
        "runner_context": "no_runner_normal",
        "zde_context": "zde_tolerance_band",
        "execution_env": "ci_batch",
    },
]

PYTEST_BASE_CMD: List[str] = ["pytest"]
TEST_TARGET: str = "tests"


@dataclass
class RunSummary:
    run_id: str
    protocol_id: str
    scenario_id: str
    data_tier: str
    timestamp_utc: str
    pytest_returncode: int
    junit_xml: str
    log_file: str
    notes: str


def _now_utc_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_dirs() -> Dict[str, Path]:
    reports_root = Path("reports") / "protocol_matrix"
    junit_root = reports_root / "raw_junit"
    summaries_root = reports_root / "summaries"
    logs_root = Path("logs") / "protocol_matrix"

    junit_root.mkdir(parents=True, exist_ok=True)
    summaries_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    return {
        "reports_root": reports_root,
        "junit_root": junit_root,
        "summaries_root": summaries_root,
        "logs_root": logs_root,
    }


def run_single(protocol_id: str, scenario: Dict[str, str], dirs: Dict[str, Path]) -> RunSummary:
    run_uuid = uuid.uuid4().hex[:8]
    timestamp = _now_utc_iso()
    run_id = f"{protocol_id}_{scenario['id']}_{run_uuid}"

    junit_path = dirs["junit_root"] / f"{run_id}.xml"
    log_path = dirs["logs_root"] / f"{run_id}.log"
    summary_path = dirs["summaries_root"] / f"{run_id}.json"

    env = os.environ.copy()
    env["APEX_PROTOCOL_ID"] = protocol_id
    env["APEX_SCENARIO_ID"] = scenario["id"]
    env["APEX_SCENARIO_JSON"] = json.dumps(scenario, separators=(",", ":"))
    env["APEX_DATA_TIER"] = "free"

    cmd = PYTEST_BASE_CMD + [
        TEST_TARGET,
        "-q",
        "--disable-warnings",
        f"--junitxml={junit_path}",
    ]

    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
        capture_output=True,
    )

    log_body = []
    log_body.append(f"=== QuantraCore Apex Protocol Scenario Run ===")
    log_body.append(f"Run ID       : {run_id}")
    log_body.append(f"Protocol     : {protocol_id}")
    log_body.append(f"Scenario     : {scenario['id']}")
    log_body.append(f"Data tier    : free (Polygon/alt free-tier OHLCV)")
    log_body.append(f"Timestamp    : {timestamp}")
    log_body.append(f"Return code  : {proc.returncode}")
    log_body.append("")
    log_body.append("----- STDOUT -----")
    log_body.append(proc.stdout)
    log_body.append("----- STDERR -----")
    log_body.append(proc.stderr)
    log_path.write_text("\n".join(log_body), encoding="utf-8")

    notes = "pytest suite executed for protocol/scenario pair; see junit_xml + log"

    summary = RunSummary(
        run_id=run_id,
        protocol_id=protocol_id,
        scenario_id=scenario["id"],
        data_tier="free",
        timestamp_utc=timestamp,
        pytest_returncode=proc.returncode,
        junit_xml=str(junit_path),
        log_file=str(log_path),
        notes=notes,
    )

    summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

    return summary


def build_manifest(all_summaries: List[RunSummary], dirs: Dict[str, Path]) -> None:
    manifest = {
        "system": "QuantraCore Apex",
        "version": "v9.0-A (Institutional Hardening)",
        "data_tier": "free",
        "description": textwrap.dedent(
            """
            Protocol Scenario Matrix — all results in this manifest were produced
            using free-tier OHLCV data (Polygon/alt APIs). These runs validate
            structural correctness and determinism only. Institutional-grade
            data revalidation will be added as additional manifests in the
            future (data_tier = "institutional") and will not overwrite these
            free-tier results.
            """
        ).strip(),
        "generated_utc": _now_utc_iso(),
        "protocols": {
            "tier": TIER_PROTOCOLS,
            "learning": LEARNING_PROTOCOLS,
            "monster": MONSTER_PROTOCOLS,
            "omega": OMEGA_PROTOCOLS,
        },
        "scenario_definitions": SCENARIOS,
        "total_runs": len(all_summaries),
        "passed": sum(1 for s in all_summaries if s.pytest_returncode == 0),
        "failed": sum(1 for s in all_summaries if s.pytest_returncode != 0),
        "runs": [asdict(s) for s in all_summaries],
    }

    manifest_path = dirs["reports_root"] / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    dirs = _ensure_dirs()
    all_summaries: List[RunSummary] = []

    total_runs = len(ALL_PROTOCOLS) * len(SCENARIOS)

    print("=" * 80)
    print("QuantraCore Apex — Full Protocol Scenario Matrix Runner")
    print("=" * 80)
    print(f"Data tier       : FREE (Polygon/alt OHLCV only)")
    print(f"Total protocols : {len(ALL_PROTOCOLS)}")
    print(f"  - Tier (T01-T80)       : {len(TIER_PROTOCOLS)}")
    print(f"  - Learning (LP01-LP25) : {len(LEARNING_PROTOCOLS)}")
    print(f"  - Monster (MR01-MR05)  : {len(MONSTER_PROTOCOLS)}")
    print(f"  - Omega (Ω1-Ω5)        : {len(OMEGA_PROTOCOLS)}")
    print(f"Total scenarios : {len(SCENARIOS)}")
    print(f"Total runs      : {total_runs}")
    print("=" * 80)
    print()

    run_count = 0
    for protocol_id, scenario in itertools.product(ALL_PROTOCOLS, SCENARIOS):
        run_count += 1
        label = f"{protocol_id} / {scenario['id']}"
        print(f"[{run_count:04d}/{total_runs}] {label}", end=" ")
        summary = run_single(protocol_id, scenario, dirs)
        all_summaries.append(summary)
        if summary.pytest_returncode != 0:
            print(f"-> WARN (exit={summary.pytest_returncode})")
        else:
            print("-> OK")

    build_manifest(all_summaries, dirs)

    passed = sum(1 for s in all_summaries if s.pytest_returncode == 0)
    failed = sum(1 for s in all_summaries if s.pytest_returncode != 0)

    print()
    print("=" * 80)
    print("PROTOCOL MATRIX SUMMARY")
    print("=" * 80)
    print(f"Total runs : {len(all_summaries)}")
    print(f"Passed     : {passed}")
    print(f"Failed     : {failed}")
    print()
    print(f"Manifest   : {dirs['reports_root'] / 'manifest.json'}")
    print(f"JUnit XML  : {dirs['junit_root']}")
    print(f"Logs       : {dirs['logs_root']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
