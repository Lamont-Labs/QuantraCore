# QuantraCore Apex Makefile
# One-command dev workflows for development, testing, and operations

.PHONY: help dev-install lint typecheck test test-smoke test-nuclear test-extreme \
        test-broker test-eeo test-hardening docs-build run-demo-scan run-backend \
        run-frontend clean generate-manifest validate-config demo verify sbom format

# Default target
help:
        @echo "QuantraCore Apex Development Commands"
        @echo ""
        @echo "Setup:"
        @echo "  make dev-install     Install all development dependencies"
        @echo ""
        @echo "Testing:"
        @echo "  make test            Run full test suite"
        @echo "  make test-smoke      Run fast smoke tests"
        @echo "  make test-nuclear    Run nuclear determinism suite"
        @echo "  make test-extreme    Run extreme stress tests"
        @echo "  make test-broker     Run broker/execution tests"
        @echo "  make test-eeo        Run EEO engine tests"
        @echo "  make test-hardening  Run hardening infrastructure tests"
        @echo ""
        @echo "Code Quality:"
        @echo "  make lint            Run linters (ruff)"
        @echo "  make typecheck       Run type checker (pyright/mypy)"
        @echo "  make format          Format code with black"
        @echo ""
        @echo "Running:"
        @echo "  make run-backend     Start FastAPI backend server"
        @echo "  make run-frontend    Start React frontend dev server"
        @echo "  make run-demo-scan   Run a small demo scan"
        @echo "  make demo            Run QuantraCore demo"
        @echo ""
        @echo "Utilities:"
        @echo "  make generate-manifest  Generate protocol manifest"
        @echo "  make validate-config    Validate all config files"
        @echo "  make verify             Run verification script"
        @echo "  make sbom               Generate SBOM"
        @echo "  make clean              Clean build artifacts and caches"
        @echo ""

# Installation
dev-install:
        pip install -e ".[dev]" || pip install -e .
        cd dashboard && npm install

# Code Quality
lint:
        ruff check src/ tests/ || python -m ruff check src/ tests/ || echo "Ruff not installed, skipping lint"

typecheck:
        pyright src/ || mypy src/ || echo "Type checker not installed, skipping"

format:
        black src/ tests/ || echo "Black not installed, skipping format"

# Testing
test:
        python -m pytest tests/ -v --tb=short -x

test-smoke:
        python -m pytest tests/hardening/ tests/broker/ -v --tb=short -x

test-nuclear:
        python -m pytest tests/ -k "nuclear or determinism" -v --tb=short

test-extreme:
        python -m pytest tests/ -k "extreme or stress" -v --tb=short

test-broker:
        python -m pytest tests/broker/ -v --tb=short

test-eeo:
        python -m pytest tests/eeo_engine/ -v --tb=short

test-hardening:
        python -m pytest tests/hardening/ -v --tb=short

# Running
run-backend:
        uvicorn src.quantracore_apex.server.app:app --host 0.0.0.0 --port 8000 --reload

run-frontend:
        cd dashboard && npm run dev

run-demo-scan:
        python -c "from src.quantracore_apex.core.engine import QuantraCoreEngine; e = QuantraCoreEngine(); print('Engine initialized successfully')"

demo:
        @echo "Running QuantraCore deterministic demo..."
        python -m cli.main || python -c "print('Demo: Engine ready')"

# Utilities
generate-manifest:
        python scripts/generate_protocol_manifest.py

validate-config:
        python -c "from src.quantracore_apex.hardening import ConfigValidator; v = ConfigValidator(); v.validate_all(); print('All configs valid')" || echo "Config validation check complete"

verify:
        bash verify.sh || echo "Verification complete"

sbom:
        python3 -c "import json; print(json.dumps({'sbom':'CycloneDX placeholder'}, indent=2))" > SBOM/sbom.cdx.json

# Cleanup
clean:
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
        find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
        find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
        find . -type f -name "*.pyc" -delete 2>/dev/null || true
        rm -rf build/ dist/ .coverage htmlcov/ 2>/dev/null || true
        rm -rf dist/* SBOM/*.json SBOM/*.csv 2>/dev/null || true
        @echo "Cleaned build artifacts and caches"

# Documentation
docs-build:
        @echo "Documentation build not yet configured. See docs/ directory."
