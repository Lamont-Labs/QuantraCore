#!/usr/bin/env python3
"""
scripts/build_scan_index.py

Index all QuantraCore Apex scan logs into a single JSON file:
    logs/scan_index.json

Covers:
    - logs/random_scan/<timestamp>/
    - logs/high_vol_smallcaps/<timestamp>/
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional


LOGS_ROOT = Path("logs")
RANDOM_SCAN_ROOT = LOGS_ROOT / "random_scan"
SMALLCAP_SCAN_ROOT = LOGS_ROOT / "high_vol_smallcaps"
INDEX_PATH = LOGS_ROOT / "scan_index.json"


@dataclass
class ScanRunSummary:
    run_type: str
    timestamp: str
    path: str
    count: Optional[int]
    successful: Optional[int]
    failed: Optional[int]
    avg_quantrascore: Optional[float]
    max_quantrascore: Optional[float]
    min_quantrascore: Optional[float]
    avg_mr_fuse_score: Optional[float]
    mode: Optional[str]
    sample_size: Optional[int]
    top_runner_symbols: List[str]
    regime_distribution: Optional[Dict[str, int]]
    bucket_distribution: Optional[Dict[str, int]]


def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_runner_symbols(summary: Dict[str, Any], limit: int = 10) -> List[str]:
    result: List[str] = []
    candidates = summary.get("runner_candidates") or summary.get("top_runner_candidates") or []
    if isinstance(candidates, list):
        for c in candidates[:limit]:
            if isinstance(c, dict):
                sym = c.get("symbol")
                if isinstance(sym, str):
                    result.append(sym)
    return result


def scan_root(root: Path, run_type: str) -> List[ScanRunSummary]:
    if not root.exists():
        return []

    results: List[ScanRunSummary] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        ts = child.name

        summary_path = child / "summary.json"
        meta_path = child / "meta.json"
        summary = load_json_if_exists(summary_path) or {}
        meta = load_json_if_exists(meta_path) or {}

        count = summary.get("total_scanned") or summary.get("count")
        successful = summary.get("successful")
        failed = summary.get("failed")
        avg_score = summary.get("avg_quantrascore")
        max_score = summary.get("max_quantrascore")
        min_score = summary.get("min_quantrascore")
        avg_fuse = summary.get("avg_mr_fuse_score")

        mode = meta.get("mode") or summary.get("mode")
        sample_size = meta.get("sample_size") or summary.get("sample_size")

        regime_dist = summary.get("regime_distribution")
        bucket_dist = summary.get("bucket_distribution")

        top_runner_symbols = extract_runner_symbols(summary)

        results.append(
            ScanRunSummary(
                run_type=run_type,
                timestamp=ts,
                path=str(child),
                count=int(count) if isinstance(count, (int, float)) else None,
                successful=int(successful) if isinstance(successful, (int, float)) else None,
                failed=int(failed) if isinstance(failed, (int, float)) else None,
                avg_quantrascore=round(float(avg_score), 2) if isinstance(avg_score, (int, float)) else None,
                max_quantrascore=round(float(max_score), 2) if isinstance(max_score, (int, float)) else None,
                min_quantrascore=round(float(min_score), 2) if isinstance(min_score, (int, float)) else None,
                avg_mr_fuse_score=round(float(avg_fuse), 2) if isinstance(avg_fuse, (int, float)) else None,
                mode=mode if isinstance(mode, str) else None,
                sample_size=int(sample_size) if isinstance(sample_size, (int, float)) else None,
                top_runner_symbols=top_runner_symbols,
                regime_distribution=regime_dist if isinstance(regime_dist, dict) else None,
                bucket_distribution=bucket_dist if isinstance(bucket_dist, dict) else None,
            )
        )

    return results


def main() -> None:
    all_runs: List[ScanRunSummary] = []

    all_runs.extend(scan_root(RANDOM_SCAN_ROOT, run_type="random"))
    all_runs.extend(scan_root(SMALLCAP_SCAN_ROOT, run_type="high_vol_smallcaps"))

    all_runs.sort(key=lambda r: (r.timestamp, r.run_type))

    total_symbols_scanned = sum(r.count or 0 for r in all_runs)
    total_successful = sum(r.successful or 0 for r in all_runs)

    index_payload: Dict[str, Any] = {
        "total_runs": len(all_runs),
        "total_symbols_scanned": total_symbols_scanned,
        "total_successful": total_successful,
        "runs": [asdict(r) for r in all_runs],
    }

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INDEX_PATH.open("w", encoding="utf-8") as f:
        json.dump(index_payload, f, indent=2, sort_keys=True)

    print(f"[INFO] Scan index written to {INDEX_PATH}")
    print(f"[INFO] Total runs indexed: {len(all_runs)}")
    print(f"[INFO] Total symbols scanned: {total_symbols_scanned}")
    print(f"[INFO] Total successful: {total_successful}")


if __name__ == "__main__":
    main()
