#!/usr/bin/env python3
"""
Generate Protocol Manifest for QuantraCore Apex.

Creates the canonical protocol manifest file with execution order and hashes.
This manifest is used to verify deterministic kernel integrity on startup.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quantracore_apex.hardening.manifest import ProtocolManifest, MANIFEST_PATH


def main():
    print("Generating QuantraCore Apex Protocol Manifest...")
    
    manifest = ProtocolManifest.generate_default()
    
    print(f"  Version: {manifest.version}")
    print(f"  Engine Snapshot ID: {manifest.engine_snapshot_id}")
    print(f"  Protocol Count: {len(manifest.protocols)}")
    print(f"    - Tier Protocols: {len([p for p in manifest.protocols if p.category == 'tier'])}")
    print(f"    - Learning Protocols: {len([p for p in manifest.protocols if p.category == 'learning'])}")
    print(f"    - Monster Runner Protocols: {len([p for p in manifest.protocols if p.category == 'monster_runner'])}")
    print(f"    - Omega Protocols: {len([p for p in manifest.protocols if p.category == 'omega'])}")
    print(f"  Manifest Hash: {manifest.hash}")
    
    manifest.save()
    print(f"\nManifest saved to: {MANIFEST_PATH}")
    print(f"Hash saved to: {MANIFEST_PATH.with_suffix('.hash')}")
    print("\nDone!")


if __name__ == "__main__":
    main()
