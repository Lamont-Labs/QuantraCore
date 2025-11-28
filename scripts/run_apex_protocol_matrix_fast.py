#!/usr/bin/env python3
"""
QuantraCore Apex™ — Fast Protocol Matrix Runner (Parallel Edition)

Runs all 115 protocols × 5 scenarios = 575 combinations using parallel execution
for maximum speed. Uses ThreadPoolExecutor for concurrent pytest runs.
"""

from __future__ import annotations

import datetime as _dt
import itertools
import json
import os
import subprocess
import textwrap
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List
import threading


TIER_PROTOCOLS: List[str] = [f"T{str(i).zfill(2)}" for i in range(1, 81)]
LEARNING_PROTOCOLS: List[str] = [f"LP{str(i).zfill(2)}" for i in range(1, 26)]
MONSTER_PROTOCOLS: List[str] = [f"MR{str(i).zfill(2)}" for i in range(1, 6)]
OMEGA_PROTOCOLS: List[str] = [f"Ω{str(i)}" for i in range(1, 6)]

ALL_PROTOCOLS: List[str] = (
    TIER_PROTOCOLS + LEARNING_PROTOCOLS + MONSTER_PROTOCOLS + OMEGA_PROTOCOLS
)

SCENARIOS: List[Dict[str, str]] = [
    {"id": "bull_trend_megacap", "market_regime": "bull_trend", "cap_tier": "mega_cap"},
    {"id": "bear_trend_midcap", "market_regime": "bear_trend", "cap_tier": "mid_cap"},
    {"id": "crash_smallcap", "market_regime": "crash_event", "cap_tier": "small_cap"},
    {"id": "parabolic_microcap", "market_regime": "parabolic_up", "cap_tier": "micro_cap"},
    {"id": "sideways_megacap", "market_regime": "sideways_range", "cap_tier": "mega_cap"},
]

MAX_WORKERS = 8
print_lock = threading.Lock()


@dataclass
class RunSummary:
    run_id: str
    protocol_id: str
    scenario_id: str
    data_tier: str
    timestamp_utc: str
    pytest_returncode: int
    notes: str


def _now_utc_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_dirs() -> Dict[str, Path]:
    reports_root = Path("reports") / "protocol_matrix_fast"
    summaries_root = reports_root / "summaries"
    summaries_root.mkdir(parents=True, exist_ok=True)
    return {"reports_root": reports_root, "summaries_root": summaries_root}


def run_single(args) -> RunSummary:
    protocol_id, scenario, dirs, idx, total = args
    run_uuid = uuid.uuid4().hex[:8]
    timestamp = _now_utc_iso()
    run_id = f"{protocol_id}_{scenario['id']}_{run_uuid}"

    env = os.environ.copy()
    env["APEX_PROTOCOL_ID"] = protocol_id
    env["APEX_SCENARIO_ID"] = scenario["id"]
    env["APEX_DATA_TIER"] = "free"

    cmd = ["python", "-m", "pytest", "tests/", "-q", "--disable-warnings", "--tb=no"]

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)

    summary = RunSummary(
        run_id=run_id,
        protocol_id=protocol_id,
        scenario_id=scenario["id"],
        data_tier="free",
        timestamp_utc=timestamp,
        pytest_returncode=proc.returncode,
        notes="parallel execution",
    )

    with print_lock:
        status = "✓" if proc.returncode == 0 else "✗"
        print(f"[{idx:04d}/{total}] {protocol_id}/{scenario['id']} {status}")

    return summary


def main() -> None:
    dirs = _ensure_dirs()
    total_runs = len(ALL_PROTOCOLS) * len(SCENARIOS)

    print("=" * 80)
    print("QuantraCore Apex — Fast Protocol Matrix (Parallel)")
    print("=" * 80)
    print(f"Protocols  : {len(ALL_PROTOCOLS)}")
    print(f"Scenarios  : {len(SCENARIOS)}")
    print(f"Total runs : {total_runs}")
    print(f"Workers    : {MAX_WORKERS}")
    print("=" * 80)

    tasks = [
        (protocol_id, scenario, dirs, idx + 1, total_runs)
        for idx, (protocol_id, scenario) in enumerate(
            itertools.product(ALL_PROTOCOLS, SCENARIOS)
        )
    ]

    all_summaries: List[RunSummary] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                summary = future.result()
                all_summaries.append(summary)
            except Exception as e:
                print(f"Error: {e}")

    passed = sum(1 for s in all_summaries if s.pytest_returncode == 0)
    failed = sum(1 for s in all_summaries if s.pytest_returncode != 0)

    manifest = {
        "system": "QuantraCore Apex",
        "version": "v9.0-A",
        "generated_utc": _now_utc_iso(),
        "total_runs": len(all_summaries),
        "passed": passed,
        "failed": failed,
        "pass_rate": f"{100*passed/len(all_summaries):.1f}%" if all_summaries else "N/A",
        "runs": [asdict(s) for s in all_summaries],
    }

    manifest_path = dirs["reports_root"] / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print()
    print("=" * 80)
    print("PROTOCOL MATRIX SUMMARY")
    print("=" * 80)
    print(f"Total : {len(all_summaries)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Rate  : {100*passed/len(all_summaries):.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
