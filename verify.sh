#!/usr/bin/env bash
set -euo pipefail

echo "==> QuantraCore: verifying deterministic artifacts"
mkdir -p SBOM dist/golden_demo_outputs

# Produce demo artifact (no subcommand)
python -m cli.main

# Hash the deterministic output
if [[ -f "dist/golden_demo_outputs/AAPL_demo.json" ]]; then
  sha256sum dist/golden_demo_outputs/AAPL_demo.json > SBOM/checksums.csv
  echo "==> Checksums written to SBOM/checksums.csv"
else
  echo "ERROR: expected artifact dist/golden_demo_outputs/AAPL_demo.json not found" >&2
  exit 1
fi

echo "==> Verification complete."
