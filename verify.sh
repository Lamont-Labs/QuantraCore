#!/usr/bin/env bash
set -euo pipefail
mkdir -p SBOM dist/golden_demo_outputs
python -m cli.main run --ticker AAPL --seed 42
sha256sum dist/golden_demo_outputs/AAPL_demo.json > SBOM/checksums.csv
echo "Wrote SBOM/checksums.csv"
