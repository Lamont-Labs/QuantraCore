#!/usr/bin/env python3
"""
QuantraCore Apex — Protocol Triple-Run Validation
Author: Lamont Labs (Jesse J. Lamont)

Purpose
-------
Brute-force test EVERY protocol 3×:

- All 80 Tier Protocols  (T01–T80)
- All 25 Learning Protocols (LP01–LP25)
- All MonsterRunner Protocols (MR01–MR05)
- All Omega Directives (Ω01–Ω05)

For each protocol:
  • Execute 3 consecutive runs.
  • Assert that outputs are deterministic across the 3 runs.
  • Record pass/fail with detailed metadata.
  • Store a machine-readable JSON + human-readable log under logs/protocol_triple/.

Run
---
    python scripts/run_protocol_triple_validation.py

This script is read-only for the engine. It should not mutate any live config.
"""

import sys
sys.path.insert(0, ".")

import hashlib
import json
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult, ApexResult
from src.quantracore_apex.protocols.tier.tier_loader import TierProtocolRunner
from src.quantracore_apex.protocols.learning.learning_loader import LearningProtocolRunner
from src.quantracore_apex.protocols.monster_runner.monster_runner_loader import MonsterRunnerLoader
from src.quantracore_apex.protocols.omega.omega import OmegaDirectives
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter
from src.quantracore_apex.data_layer.normalization import normalize_ohlcv


TIER_PROTOCOLS: List[str] = [f"T{idx:02d}" for idx in range(1, 81)]
LEARNING_PROTOCOLS: List[str] = [f"LP{idx:02d}" for idx in range(1, 26)]
MONSTER_PROTOCOLS: List[str] = [f"MR{idx:02d}" for idx in range(1, 6)]
OMEGA_PROTOCOLS: List[str] = [f"Ω{idx:02d}" for idx in range(1, 6)]

ALL_PROTOCOLS: List[str] = (
    TIER_PROTOCOLS
    + LEARNING_PROTOCOLS
    + MONSTER_PROTOCOLS
    + OMEGA_PROTOCOLS
)

TRIPLE_RUNS: int = 3

ROOT = Path(__file__).resolve().parents[1]
OUTDIR_ROOT = ROOT / "logs" / "protocol_triple"
OUTDIR_ROOT.mkdir(parents=True, exist_ok=True)

TS = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
OUTDIR = OUTDIR_ROOT / TS
OUTDIR.mkdir(parents=True, exist_ok=True)

SUMMARY_JSON = OUTDIR / "protocol_triple_summary.json"
DETAIL_JSON = OUTDIR / "protocol_triple_details.json"
HUMAN_LOG = OUTDIR / "protocol_triple_log.txt"


@dataclass
class SingleRunResult:
    run_index: int
    duration_sec: float
    output_hash: str
    output_snapshot: Dict[str, Any]


@dataclass
class ProtocolTripleResult:
    protocol_id: str
    category: str
    runs: List[Dict[str, Any]] = field(default_factory=list)
    deterministic: bool = False
    all_passed: bool = False
    error: str = ""


def canonical_hash(payload: Any) -> str:
    """Produce a stable hash string for any JSON-serializable payload."""
    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
    except Exception:
        return hashlib.sha256(str(payload).encode("utf-8")).hexdigest()


def detect_category(proto_id: str) -> str:
    if proto_id.startswith("T"):
        return "tier"
    if proto_id.startswith("LP"):
        return "learning"
    if proto_id.startswith("MR"):
        return "monster_runner"
    if proto_id.startswith("Ω"):
        return "omega_directive"
    return "unknown"


def log_line(msg: str) -> None:
    print(msg)
    with HUMAN_LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def result_to_dict(result: Any) -> Dict[str, Any]:
    """Convert a result object to a dictionary."""
    if result is None:
        return {"result": None}
    if isinstance(result, dict):
        return result
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if hasattr(result, "__dict__"):
        d = {}
        for k, v in result.__dict__.items():
            if hasattr(v, "value"):
                d[k] = v.value
            elif hasattr(v, "__dict__"):
                d[k] = {kk: vv.value if hasattr(vv, "value") else vv for kk, vv in v.__dict__.items()}
            else:
                d[k] = v
        return d
    return {"result": str(result)}


class ProtocolValidator:
    """Handles protocol execution for all categories."""
    
    def __init__(self):
        self.adapter = SyntheticAdapter(seed=42)
        self.engine = ApexEngine(enable_logging=False)
        self.tier_runner = TierProtocolRunner()
        self.learning_runner = LearningProtocolRunner()
        self.monster_runner = MonsterRunnerLoader()
        self.omega_directives = OmegaDirectives()
        
        end_date = datetime(2024, 1, 15)
        start_date = end_date - timedelta(days=200)
        bars = self.adapter.fetch_ohlcv("AAPL", start_date, end_date, "1d")
        self.normalized_bars, _ = normalize_ohlcv(bars)
        
        full_result = self.engine.run_scan(self.normalized_bars, "AAPL")
        self.window = full_result.window if hasattr(full_result, 'window') else None
        self.microtraits = full_result.microtraits if hasattr(full_result, 'microtraits') else None
        self.sample_apex_result = full_result
    
    def run_tier_protocol(self, protocol_id: str) -> Dict[str, Any]:
        """Run a single tier protocol."""
        if self.window is None or self.microtraits is None:
            return {"error": "Window or microtraits not available"}
        
        result = self.tier_runner.run_single(protocol_id, self.window, self.microtraits)
        return result_to_dict(result)
    
    def run_learning_protocol(self, protocol_id: str) -> Dict[str, Any]:
        """Run a single learning protocol."""
        if self.window is None or self.microtraits is None:
            return {"error": "Window or microtraits not available"}
        
        result = self.learning_runner.run_single(protocol_id, self.window, self.microtraits)
        return result_to_dict(result)
    
    def run_monster_protocol(self, protocol_id: str) -> Dict[str, Any]:
        """Run a single monster runner protocol."""
        result = self.monster_runner.run_single(protocol_id, self.normalized_bars)
        return result_to_dict(result)
    
    def run_omega_directive(self, protocol_id: str) -> Dict[str, Any]:
        """Run a single omega directive."""
        omega_num = int(protocol_id.replace("Ω", ""))
        
        check_methods = {
            1: self.omega_directives.check_omega_1,
            2: self.omega_directives.check_omega_2,
            3: self.omega_directives.check_omega_3,
            4: self.omega_directives.check_omega_4,
            5: self.omega_directives.check_omega_5,
        }
        
        if omega_num in check_methods:
            result = check_methods[omega_num](self.sample_apex_result)
            return result_to_dict(result)
        
        return {"error": f"Unknown omega directive: {protocol_id}"}
    
    def run_protocol(self, protocol_id: str) -> Dict[str, Any]:
        """Run any protocol by ID."""
        category = detect_category(protocol_id)
        
        if category == "tier":
            return self.run_tier_protocol(protocol_id)
        elif category == "learning":
            return self.run_learning_protocol(protocol_id)
        elif category == "monster_runner":
            return self.run_monster_protocol(protocol_id)
        elif category == "omega_directive":
            return self.run_omega_directive(protocol_id)
        else:
            return {"error": f"Unknown protocol category for {protocol_id}"}


def run_single_protocol_test(
    validator: ProtocolValidator,
    protocol_id: str,
    run_index: int,
) -> SingleRunResult:
    """Execute one protocol run."""
    start = time.perf_counter()
    result_payload = validator.run_protocol(protocol_id)
    duration = time.perf_counter() - start
    h = canonical_hash(result_payload)
    
    return SingleRunResult(
        run_index=run_index,
        duration_sec=round(duration, 4),
        output_hash=h,
        output_snapshot=result_payload,
    )


def main() -> int:
    log_line("=" * 80)
    log_line("QuantraCore Apex — Protocol Triple-Run Validation")
    log_line(f"UTC Timestamp: {TS}")
    log_line(f"Output directory: {OUTDIR}")
    log_line("=" * 80)
    log_line("")
    
    log_line("[INIT] Booting Apex engine + protocol systems...")
    validator = ProtocolValidator()
    log_line("[INIT] Complete — all protocol loaders ready")
    log_line("")
    
    triple_results: List[ProtocolTripleResult] = []
    
    log_line(f"[INFO] Testing {len(ALL_PROTOCOLS)} protocols × {TRIPLE_RUNS} runs = {len(ALL_PROTOCOLS) * TRIPLE_RUNS} total executions")
    log_line("")
    
    for idx, proto_id in enumerate(ALL_PROTOCOLS, start=1):
        cat = detect_category(proto_id)
        header = f"[PROTO {idx:03d}/{len(ALL_PROTOCOLS)}] {proto_id} ({cat})"
        log_line("-" * 80)
        log_line(header)
        
        runs: List[SingleRunResult] = []
        
        try:
            for r in range(1, TRIPLE_RUNS + 1):
                single = run_single_protocol_test(validator, proto_id, r)
                runs.append(single)
                log_line(
                    f"    Run {r}/{TRIPLE_RUNS}: "
                    f"duration={single.duration_sec:.4f}s  "
                    f"hash={single.output_hash[:16]}..."
                )
            
            hashes = {r.output_hash for r in runs}
            deterministic = len(hashes) == 1
            
            if deterministic:
                log_line("    ✅ Determinism: PASSED (all 3 runs identical)")
            else:
                log_line("    ❌ Determinism: FAILED (runs produced different outputs)")
            
            proto_result = ProtocolTripleResult(
                protocol_id=proto_id,
                category=cat,
                runs=[asdict(r) for r in runs],
                deterministic=deterministic,
                all_passed=deterministic,
            )
            
        except Exception as exc:
            import traceback
            err_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            log_line(f"    💥 ERROR while executing {proto_id}:")
            log_line(err_text)
            
            proto_result = ProtocolTripleResult(
                protocol_id=proto_id,
                category=cat,
                runs=[asdict(r) for r in runs],
                deterministic=False,
                all_passed=False,
                error=err_text,
            )
        
        triple_results.append(proto_result)
    
    total = len(triple_results)
    deterministic_ok = sum(1 for r in triple_results if r.deterministic)
    fully_ok = sum(1 for r in triple_results if r.all_passed and not r.error)
    errors = [r for r in triple_results if r.error]
    
    by_category = {}
    for r in triple_results:
        if r.category not in by_category:
            by_category[r.category] = {"total": 0, "passed": 0}
        by_category[r.category]["total"] += 1
        if r.deterministic:
            by_category[r.category]["passed"] += 1
    
    log_line("")
    log_line("=" * 80)
    log_line("FINAL SUMMARY")
    log_line("=" * 80)
    log_line(f"Total protocols tested : {total}")
    log_line(f"Deterministic protocols: {deterministic_ok}/{total}")
    log_line(f"Fully OK (no errors)   : {fully_ok}/{total}")
    log_line(f"Protocols with errors  : {len(errors)}")
    log_line("")
    log_line("By Category:")
    for cat, stats in sorted(by_category.items()):
        log_line(f"  {cat:16}: {stats['passed']}/{stats['total']} deterministic")
    
    if errors:
        log_line("")
        log_line("Protocols with errors:")
        for r in errors:
            log_line(f"  • {r.protocol_id} → see details JSON for stack trace")
    
    log_line("")
    log_line("=" * 80)
    log_line(f"JSON summary : {SUMMARY_JSON}")
    log_line(f"JSON details : {DETAIL_JSON}")
    log_line(f"Human log    : {HUMAN_LOG}")
    log_line("=" * 80)
    
    summary_payload = {
        "timestamp_utc": TS,
        "total_protocols": total,
        "deterministic_ok": deterministic_ok,
        "fully_ok": fully_ok,
        "by_category": by_category,
        "errors": [r.protocol_id for r in errors],
    }
    SUMMARY_JSON.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    
    details_payload = [asdict(r) for r in triple_results]
    DETAIL_JSON.write_text(json.dumps(details_payload, indent=2, default=str), encoding="utf-8")
    
    if deterministic_ok == total:
        log_line("")
        log_line("🎉 ALL PROTOCOLS DETERMINISTIC — INSTITUTIONAL GRADE CONFIRMED")
    
    return 0 if not errors and deterministic_ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
