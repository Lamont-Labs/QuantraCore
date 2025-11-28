#!/usr/bin/env python3
"""
QuantraCore Apex™ — Chunked Test Runner (Replit-safe)

Goal
----
Run *all* test scripts and pytest suites in smaller chunks to reduce the risk
of Replit / CI timeouts, while still exercising the full system.

What it does
------------
1. Finds and runs all Python test scripts in `scripts/` that match `run_*.py`.
2. Breaks the pytest suite into multiple chunks:
   - One chunk per top-level subdirectory in `tests/`
   - One chunk per root-level `test_*.py` file in `tests/`
3. Runs each chunk sequentially, logging output per chunk under:
   - logs/chunked_tests/

4. Writes an overall summary JSON + TXT to:
   - reports/chunked_tests/

Usage
-----
From Replit shell or CI:

    python scripts/apex_chunked_test_runner.py

Options:

    --dry-run
        Show which chunks would be run and where logs would go, but don't execute.

    --no-scripts
        Do not run scripts/run_*.py chunks.

    --no-pytest
        Do not run pytest-based chunks (tests/).
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
from typing import List, Dict


@dataclass
class Chunk:
    name: str
    kind: str
    command: List[str]
    workdir: str


@dataclass
class ChunkResult:
    name: str
    kind: str
    command: List[str]
    started_utc: str
    finished_utc: str
    return_code: int
    log_path: str


@dataclass
class ChunkedRunSummary:
    system: str
    version: str
    generated_utc: str
    overall_status: str
    chunks: List[ChunkResult]


def _utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_dirs() -> Dict[str, Path]:
    logs_root = Path("logs") / "chunked_tests"
    reports_root = Path("reports") / "chunked_tests"
    logs_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    return {"logs_root": logs_root, "reports_root": reports_root}


def _discover_script_chunks() -> List[Chunk]:
    chunks: List[Chunk] = []
    scripts_dir = Path("scripts")
    if not scripts_dir.is_dir():
        return chunks

    for path in sorted(scripts_dir.glob("run_*.py")):
        chunks.append(
            Chunk(
                name=f"script_{path.stem}",
                kind="script",
                command=[sys.executable, str(path)],
                workdir=str(Path(".")),
            )
        )
    return chunks


def _discover_pytest_chunks() -> List[Chunk]:
    chunks: List[Chunk] = []
    tests_dir = Path("tests")
    if not tests_dir.is_dir():
        return chunks

    for test_file in sorted(
        p for p in tests_dir.iterdir()
        if p.is_file() and p.name.startswith("test_") and p.suffix == ".py"
    ):
        chunks.append(
            Chunk(
                name=f"pytest_file_{test_file.name}",
                kind="pytest",
                command=["pytest", str(test_file), "--disable-warnings", "--maxfail=1"],
                workdir=str(Path(".")),
            )
        )

    for subdir in sorted(p for p in tests_dir.iterdir() if p.is_dir() and not p.name.startswith("__")):
        chunks.append(
            Chunk(
                name=f"pytest_dir_{subdir.name}",
                kind="pytest",
                command=["pytest", str(subdir), "--disable-warnings", "--maxfail=1"],
                workdir=str(Path(".")),
            )
        )

    return chunks


def _run_chunk(chunk: Chunk, logs_root: Path, dry_run: bool = False) -> ChunkResult:
    started = _utc_now_iso()
    log_file = logs_root / f"{chunk.name}_{_dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"

    if dry_run:
        body = [
            f"[DRY-RUN] Chunk: {chunk.name}",
            f"Kind      : {chunk.kind}",
            f"Command   : {' '.join(chunk.command)}",
            f"Workdir   : {chunk.workdir}",
        ]
        log_file.write_text("\n".join(body), encoding="utf-8")
        finished = _utc_now_iso()
        return ChunkResult(
            name=chunk.name,
            kind=chunk.kind,
            command=chunk.command,
            started_utc=started,
            finished_utc=finished,
            return_code=0,
            log_path=str(log_file),
        )

    proc = subprocess.run(chunk.command, cwd=chunk.workdir, text=True, capture_output=True)
    finished = _utc_now_iso()

    body = [
        f"Chunk: {chunk.name}",
        f"Kind       : {chunk.kind}",
        f"Command    : {' '.join(chunk.command)}",
        f"Workdir    : {chunk.workdir}",
        f"Started UTC: {started}",
        f"Finished UTC: {finished}",
        f"Return code: {proc.returncode}",
        "",
        "----- STDOUT -----",
        proc.stdout,
        "----- STDERR -----",
        proc.stderr,
    ]
    log_file.write_text("\n".join(body), encoding="utf-8")

    return ChunkResult(
        name=chunk.name,
        kind=chunk.kind,
        command=chunk.command,
        started_utc=started,
        finished_utc=finished,
        return_code=proc.returncode,
        log_path=str(log_file),
    )


def _write_summary(results: List[ChunkResult], dirs: Dict[str, Path]) -> None:
    overall = "OK"
    for r in results:
        if r.return_code != 0:
            overall = "FAILED"
            break

    summary = ChunkedRunSummary(
        system="QuantraCore Apex",
        version="v9.0-A (Institutional Hardening)",
        generated_utc=_utc_now_iso(),
        overall_status=overall,
        chunks=results,
    )

    summary_json = dirs["reports_root"] / "chunked_test_run_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "system": summary.system,
                "version": summary.version,
                "generated_utc": summary.generated_utc,
                "overall_status": summary.overall_status,
                "chunks": [asdict(r) for r in summary.chunks],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_txt = dirs["reports_root"] / "chunked_test_run_summary.txt"
    lines: List[str] = [
        "QuantraCore Apex — Chunked Test Run Summary",
        "=" * 50,
        f"Generated UTC : {summary.generated_utc}",
        f"System        : {summary.system}",
        f"Version       : {summary.version}",
        f"Overall       : {summary.overall_status}",
        "",
        "Chunks:",
    ]
    for r in summary.chunks:
        lines.extend([
            f"- {r.name}",
            f"  Kind        : {r.kind}",
            f"  Command     : {' '.join(r.command)}",
            f"  Started UTC : {r.started_utc}",
            f"  Finished UTC: {r.finished_utc}",
            f"  Return code : {r.return_code}",
            f"  Log         : {r.log_path}",
            "",
        ])
    summary_txt.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QuantraCore Apex — Chunked Test Runner (Replit-safe)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show planned chunks without executing.")
    parser.add_argument("--no-scripts", action="store_true", help="Do not run scripts/run_*.py chunks.")
    parser.add_argument("--no-pytest", action="store_true", help="Do not run pytest-based chunks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dirs = _ensure_dirs()

    print("=" * 60)
    print("QuantraCore Apex — Chunked Test Runner")
    print("=" * 60)
    print(f"Dry run   : {args.dry_run}")
    print(f"Logs dir  : {dirs['logs_root']}")
    print(f"Reports   : {dirs['reports_root']}")
    print()

    chunks: List[Chunk] = []

    if not args.no_scripts:
        script_chunks = _discover_script_chunks()
        if script_chunks:
            print(f"Discovered {len(script_chunks)} script chunks.")
            chunks.extend(script_chunks)
        else:
            print("No scripts/run_*.py found — no script chunks.")
    else:
        print("Skipping script chunks (--no-scripts).")

    if not args.no_pytest:
        pytest_chunks = _discover_pytest_chunks()
        if pytest_chunks:
            print(f"Discovered {len(pytest_chunks)} pytest chunks.")
            chunks.extend(pytest_chunks)
        else:
            print("No tests/ directory or no test_*.py files/subdirs — no pytest chunks.")
    else:
        print("Skipping pytest chunks (--no-pytest).")

    if not chunks:
        print("No test chunks discovered. Exiting.")
        return

    print()
    print("Planned chunks:")
    for c in chunks:
        print(f"  - {c.name}: {' '.join(c.command)}")
    print()

    results: List[ChunkResult] = []
    for c in chunks:
        print(f"[RUN] {c.name}")
        res = _run_chunk(c, logs_root=dirs["logs_root"], dry_run=args.dry_run)
        results.append(res)
        if res.return_code != 0:
            print(f"  -> WARNING: chunk returned non-zero code: {res.return_code}")
        else:
            print("  -> OK")

    _write_summary(results, dirs)

    print()
    print("-" * 60)
    print("Chunked test run completed.")
    print(f"Overall status: {'FAILED' if any(r.return_code != 0 for r in results) else 'OK'}")


if __name__ == "__main__":
    main()
