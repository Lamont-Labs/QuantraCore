#!/usr/bin/env python3
"""
QuantraCore Apex™ — Institutional Bootstrap & Hardware-Aware Scaling Script

Purpose
-------
Make the entire QuantraCore Apex system as "plug-and-play" as possible for
institutional environments:

  - Auto-detect hardware profile (cores, RAM, GPUs if any).
  - Derive a hardware tier (SMALL / MEDIUM / LARGE / XL / CLUSTER_HINT).
  - Generate a runtime config describing:
        * scanner scaling
        * engine worker counts
        * ApexLab training parallelism
        * ApexCore model selection (Big vs Mini)
        * UI concurrency hints
        * logging / retention defaults
  - Mark results explicitly as "institution-ready" but data-tier-agnostic:
        * free-tier compatible
        * can be re-run with institutional data feeds
  - Emit human-readable + machine-readable artifacts into `config/` and `reports/`.

Usage (institutional)
---------------------
  $ python scripts/apex_institutional_bootstrap.py
    - Profiles current machine
    - Writes:
        config/apex_institutional_profile.json
        config/apex_institutional_profile.yaml
        reports/institutional_bootstrap/<timestamp>/bootstrap_report.txt

Optional flags:
  --dry-run           : only show decisions, do not write files
  --no-gpu-detect     : skip GPU detection (for locked-down environments)
  --profile-only      : profile hardware, no runtime scaling suggestions
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import psutil
except ImportError:
    psutil = None

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class HardwareProfile:
    hostname: str
    os: str
    os_version: str
    python_version: str
    cpu_model: str
    cpu_cores_logical: int
    cpu_cores_physical: int
    ram_gb: float
    gpu_present: bool
    gpu_name: Optional[str]
    gpu_count: int


@dataclass
class ScalingProfile:
    tier: str
    tier_reason: str
    scanner_max_symbols: int
    scanner_batch_size: int
    scanner_parallel_jobs: int
    engine_workers: int
    engine_max_concurrent_signals: int
    apexlab_training_workers: int
    apexlab_max_parallel_jobs: int
    apexlab_recommended_checkpoint_interval: int
    apexcore_variant: str
    apexcore_batch_size: int
    ui_max_concurrent_sessions: int
    ui_enable_server_side_render: bool
    log_retention_days: int
    log_max_bytes_per_file: int


@dataclass
class InstitutionalBootstrapConfig:
    system: str
    version: str
    generated_utc: str
    data_tier_note: str
    hardware: HardwareProfile
    scaling: ScalingProfile
    environment_hints: Dict[str, Any]


def _detect_gpu(no_gpu_detect: bool = False) -> Tuple[bool, Optional[str], int]:
    if no_gpu_detect:
        return False, None, 0

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            out = subprocess.check_output(
                [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            if lines:
                return True, lines[0], len(lines)
        except Exception:
            pass

    return False, None, 0


def profile_hardware(no_gpu_detect: bool = False) -> HardwareProfile:
    hostname = platform.node() or "unknown-host"
    os_name = platform.system()
    os_version = platform.version()
    python_version = platform.python_version()
    cpu_model = platform.processor() or "unknown-cpu"

    if psutil is not None:
        cpu_cores_logical = psutil.cpu_count(logical=True) or os.cpu_count() or 1
        cpu_cores_physical = psutil.cpu_count(logical=False) or cpu_cores_logical
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    else:
        cpu_cores_logical = os.cpu_count() or 1
        cpu_cores_physical = cpu_cores_logical
        ram_gb = 8.0

    gpu_present, gpu_name, gpu_count = _detect_gpu(no_gpu_detect=no_gpu_detect)

    return HardwareProfile(
        hostname=hostname,
        os=os_name,
        os_version=os_version,
        python_version=python_version,
        cpu_model=cpu_model,
        cpu_cores_logical=cpu_cores_logical,
        cpu_cores_physical=cpu_cores_physical,
        ram_gb=round(ram_gb, 2),
        gpu_present=gpu_present,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
    )


def derive_scaling(h: HardwareProfile) -> ScalingProfile:
    cores = h.cpu_cores_logical
    ram = h.ram_gb

    if cores <= 4 or ram <= 16:
        tier = "SMALL"
        tier_reason = "<=4 logical cores or <=16GB RAM — tuned for workstation / dev nodes."
    elif cores <= 8 or ram <= 32:
        tier = "MEDIUM"
        tier_reason = "<=8 logical cores or <=32GB RAM — tuned for single-node institutional pilot."
    elif cores <= 16 or ram <= 64:
        tier = "LARGE"
        tier_reason = "<=16 logical cores or <=64GB RAM — tuned for production node."
    else:
        tier = "XL"
        tier_reason = ">16 logical cores or >64GB RAM — tuned for heavy institutional loads."

    if tier == "SMALL":
        scanner_max_symbols = 2000
        scanner_batch_size = 200
        scanner_parallel_jobs = max(1, cores // 2)
        engine_workers = max(1, cores // 2)
        engine_max_concurrent_signals = 500
        apexlab_training_workers = max(1, cores // 4)
        apexlab_max_parallel_jobs = max(1, cores // 4)
        checkpoint_interval = 60
        apexcore_variant = "mini"
        apexcore_batch_size = 128
        ui_max_concurrent_sessions = 5
        ui_ssr = False
        log_retention_days = 7
        log_max_bytes = 50 * 1024 * 1024
    elif tier == "MEDIUM":
        scanner_max_symbols = 5000
        scanner_batch_size = 500
        scanner_parallel_jobs = max(2, cores // 2)
        engine_workers = max(2, cores - 2)
        engine_max_concurrent_signals = 2000
        apexlab_training_workers = max(2, cores // 3)
        apexlab_max_parallel_jobs = max(2, cores // 3)
        checkpoint_interval = 45
        apexcore_variant = "big"
        apexcore_batch_size = 256
        ui_max_concurrent_sessions = 15
        ui_ssr = True
        log_retention_days = 14
        log_max_bytes = 200 * 1024 * 1024
    elif tier == "LARGE":
        scanner_max_symbols = 15000
        scanner_batch_size = 1000
        scanner_parallel_jobs = max(4, cores // 2)
        engine_workers = max(4, cores - 2)
        engine_max_concurrent_signals = 5000
        apexlab_training_workers = max(4, cores // 2)
        apexlab_max_parallel_jobs = max(4, cores // 2)
        checkpoint_interval = 30
        apexcore_variant = "hybrid" if h.gpu_present else "big"
        apexcore_batch_size = 512 if not h.gpu_present else 1024
        ui_max_concurrent_sessions = 50
        ui_ssr = True
        log_retention_days = 30
        log_max_bytes = 500 * 1024 * 1024
    else:
        scanner_max_symbols = 50000
        scanner_batch_size = 2000
        scanner_parallel_jobs = max(8, cores // 2)
        engine_workers = max(8, cores - 4)
        engine_max_concurrent_signals = 20000
        apexlab_training_workers = max(8, cores // 2)
        apexlab_max_parallel_jobs = max(8, cores // 2)
        checkpoint_interval = 20
        apexcore_variant = "hybrid" if h.gpu_present else "big"
        apexcore_batch_size = 1024
        ui_max_concurrent_sessions = 150
        ui_ssr = True
        log_retention_days = 90
        log_max_bytes = 2 * 1024 * 1024 * 1024

    return ScalingProfile(
        tier=tier,
        tier_reason=tier_reason,
        scanner_max_symbols=scanner_max_symbols,
        scanner_batch_size=scanner_batch_size,
        scanner_parallel_jobs=scanner_parallel_jobs,
        engine_workers=engine_workers,
        engine_max_concurrent_signals=engine_max_concurrent_signals,
        apexlab_training_workers=apexlab_training_workers,
        apexlab_max_parallel_jobs=apexlab_max_parallel_jobs,
        apexlab_recommended_checkpoint_interval=checkpoint_interval,
        apexcore_variant=apexcore_variant,
        apexcore_batch_size=apexcore_batch_size,
        ui_max_concurrent_sessions=ui_max_concurrent_sessions,
        ui_enable_server_side_render=ui_ssr,
        log_retention_days=log_retention_days,
        log_max_bytes_per_file=log_max_bytes,
    )


def _utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_dirs() -> Dict[str, Path]:
    config_root = Path("config")
    reports_root = Path("reports") / "institutional_bootstrap"
    config_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    return {"config_root": config_root, "reports_root": reports_root}


def write_config_files(cfg: InstitutionalBootstrapConfig, dirs: Dict[str, Path]) -> Dict[str, Path]:
    config_root = dirs["config_root"]
    ts_slug = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    json_path = config_root / "apex_institutional_profile.json"
    yaml_path = config_root / "apex_institutional_profile.yaml"

    json_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    if yaml is not None:
        yaml_path.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")

    report_path = dirs["reports_root"] / f"bootstrap_report_{ts_slug}.txt"
    report_lines = [
        "QuantraCore Apex — Institutional Bootstrap Report",
        "=" * 60,
        f"Generated UTC      : {cfg.generated_utc}",
        f"System             : {cfg.system}",
        f"Version            : {cfg.version}",
        "",
        "Hardware Profile",
        "-" * 40,
        f"Hostname           : {cfg.hardware.hostname}",
        f"OS                 : {cfg.hardware.os} {cfg.hardware.os_version}",
        f"Python             : {cfg.hardware.python_version}",
        f"CPU model          : {cfg.hardware.cpu_model}",
        f"Logical cores      : {cfg.hardware.cpu_cores_logical}",
        f"Physical cores     : {cfg.hardware.cpu_cores_physical}",
        f"RAM (GB)           : {cfg.hardware.ram_gb}",
        f"GPU present        : {cfg.hardware.gpu_present}",
        f"GPU name           : {cfg.hardware.gpu_name or '-'}",
        f"GPU count          : {cfg.hardware.gpu_count}",
        "",
        "Scaling Profile",
        "-" * 40,
        f"Tier               : {cfg.scaling.tier}",
        f"Tier reason        : {cfg.scaling.tier_reason}",
        "",
        "Scanner",
        f"  max_symbols      : {cfg.scaling.scanner_max_symbols}",
        f"  batch_size       : {cfg.scaling.scanner_batch_size}",
        f"  parallel_jobs    : {cfg.scaling.scanner_parallel_jobs}",
        "",
        "Engine",
        f"  workers          : {cfg.scaling.engine_workers}",
        f"  max_concurrent   : {cfg.scaling.engine_max_concurrent_signals}",
        "",
        "ApexLab",
        f"  training_workers : {cfg.scaling.apexlab_training_workers}",
        f"  max_parallel     : {cfg.scaling.apexlab_max_parallel_jobs}",
        f"  ckpt_interval    : {cfg.scaling.apexlab_recommended_checkpoint_interval} min",
        "",
        "ApexCore",
        f"  variant          : {cfg.scaling.apexcore_variant}",
        f"  batch_size       : {cfg.scaling.apexcore_batch_size}",
        "",
        "UI",
        f"  max_sessions     : {cfg.scaling.ui_max_concurrent_sessions}",
        f"  SSR enabled      : {cfg.scaling.ui_enable_server_side_render}",
        "",
        "Logging",
        f"  retention_days   : {cfg.scaling.log_retention_days}",
        f"  max_bytes/file   : {cfg.scaling.log_max_bytes_per_file}",
        "",
        "Data Tier Note",
        "-" * 40,
        cfg.data_tier_note,
    ]

    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {"json": json_path, "yaml": yaml_path, "report": report_path}


def build_env_hints(h: HardwareProfile, s: ScalingProfile) -> Dict[str, Any]:
    return {
        "recommended_env": {
            "APEX_ENGINE_WORKERS": s.engine_workers,
            "APEX_ENGINE_MAX_CONCURRENT_SIGNALS": s.engine_max_concurrent_signals,
            "APEX_SCANNER_MAX_SYMBOLS": s.scanner_max_symbols,
            "APEX_SCANNER_BATCH_SIZE": s.scanner_batch_size,
            "APEX_SCANNER_PARALLEL_JOBS": s.scanner_parallel_jobs,
            "APEXLAB_TRAINING_WORKERS": s.apexlab_training_workers,
            "APEXLAB_MAX_PARALLEL_JOBS": s.apexlab_max_parallel_jobs,
            "APEXLAB_CKPT_INTERVAL_MIN": s.apexlab_recommended_checkpoint_interval,
            "APEXCORE_VARIANT": s.apexcore_variant,
            "APEXCORE_BATCH_SIZE": s.apexcore_batch_size,
            "APEX_UI_MAX_SESSIONS": s.ui_max_concurrent_sessions,
            "APEX_UI_ENABLE_SSR": int(s.ui_enable_server_side_render),
            "APEX_LOG_RETENTION_DAYS": s.log_retention_days,
            "APEX_LOG_MAX_BYTES": s.log_max_bytes_per_file,
        },
        "integration_notes": [
            "These values are safe defaults derived from detected hardware.",
            "Institutions may increase limits gradually while monitoring CPU/RAM.",
            "For clustered deployments, run this script once per node and merge profiles.",
            "All settings are compatible with both free-tier and institutional data feeds.",
        ],
        "hardware_summary": asdict(h),
        "scaling_summary": asdict(s),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuantraCore Apex — Institutional Bootstrap")
    parser.add_argument("--dry-run", action="store_true", help="Profile but do not write files.")
    parser.add_argument("--no-gpu-detect", action="store_true", help="Skip GPU detection.")
    parser.add_argument("--profile-only", action="store_true", help="Only profile hardware.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("QuantraCore Apex — Institutional Bootstrap")
    print("=" * 60)

    hw = profile_hardware(no_gpu_detect=args.no_gpu_detect)
    sc = derive_scaling(hw)

    print(f"Hostname           : {hw.hostname}")
    print(f"OS                 : {hw.os} {hw.os_version}")
    print(f"Python             : {hw.python_version}")
    print(f"CPU model          : {hw.cpu_model}")
    print(f"Logical cores      : {hw.cpu_cores_logical}")
    print(f"Physical cores     : {hw.cpu_cores_physical}")
    print(f"RAM (GB)           : {hw.ram_gb}")
    print(f"GPU present        : {hw.gpu_present} (name={hw.gpu_name}, count={hw.gpu_count})")
    print()
    print(f"Derived tier       : {sc.tier}")
    print(f"Tier reason        : {sc.tier_reason}")
    print()
    print(f"Scanner: max={sc.scanner_max_symbols}, batch={sc.scanner_batch_size}, jobs={sc.scanner_parallel_jobs}")
    print(f"Engine: workers={sc.engine_workers}, max_concurrent={sc.engine_max_concurrent_signals}")
    print(f"ApexLab: workers={sc.apexlab_training_workers}, parallel={sc.apexlab_max_parallel_jobs}")
    print(f"ApexCore: variant={sc.apexcore_variant}, batch={sc.apexcore_batch_size}")
    print(f"UI: sessions={sc.ui_max_concurrent_sessions}, SSR={sc.ui_enable_server_side_render}")
    print()

    if args.profile_only:
        print("Profile-only mode: not writing configs.")
        return

    dirs = _ensure_dirs()
    env_hints = build_env_hints(hw, sc)

    cfg = InstitutionalBootstrapConfig(
        system="QuantraCore Apex",
        version="v9.0-A (Institutional Hardening)",
        generated_utc=_utc_now_iso(),
        data_tier_note=(
            "This bootstrap profile is data-tier agnostic. It is compatible with both "
            "free-tier OHLCV data (e.g., Polygon free) and institutional data feeds. "
            "Institutions are expected to wire their own feeds into the scanner layer; "
            "this config only dictates hardware-aware scaling heuristics."
        ),
        hardware=hw,
        scaling=sc,
        environment_hints=env_hints,
    )

    if args.dry_run:
        print("Dry-run mode: configs NOT written.")
        return

    paths = write_config_files(cfg, dirs)
    print("Configs written:")
    print(f"  JSON   : {paths['json']}")
    print(f"  YAML   : {paths['yaml']}" if yaml else "  YAML   : (skipped - PyYAML not installed)")
    print(f"  Report : {paths['report']}")
    print()
    print("Institutional bootstrap complete.")


if __name__ == "__main__":
    main()
