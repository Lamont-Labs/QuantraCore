# Developer Guide

**Version:** 8.0  
**Audience:** Engineers onboarding to QuantraCore Apex development

---

## 1. Welcome

This guide provides engineers with the essential information needed to understand, navigate, and contribute to the QuantraCore Apex codebase. It covers architecture, directory structure, development workflows, and conventions.

---

## 2. High-Level Architecture

QuantraCore Apex is a hybrid AI trading intelligence system with three main architectural layers:

### 2.1 Deterministic Core (Apex Engine)

The heart of the system—a fully deterministic rule-based engine that:
- Processes market data through a defined pipeline
- Produces reproducible outputs for any given input
- Serves as the "teacher" for neural models
- Enforces safety through Omega directives

### 2.2 Neural Intelligence (ApexCore Models)

Trainable models that approximate the Apex engine:
- **ApexCore Full** — Desktop-class model (4–20MB)
- **ApexCore Mini** — Mobile-optimized model (0.5–3MB)
- Trained offline by ApexLab
- Aligned with Apex through distillation

### 2.3 Application Layer

User-facing components:
- **Apex Dashboard** — React-based visualization console
- **QuantraVision Apex** — Mobile overlay copilot
- **CLI** — Command-line interface for testing and demos
- **API** — FastAPI endpoints for integration

---

## 3. Directory Map

```
QuantraCore/
├── README.md                 # Project overview
├── INSTALL.md                # Installation instructions
├── CHANGELOG.md              # Version history
├── Makefile                  # Build and run commands
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Python project configuration
├── verify.sh                 # Determinism verification script
│
├── src/                      # Source code
│   ├── api/                  # FastAPI application
│   │   ├── __init__.py
│   │   ├── asgi.py           # ASGI entry point
│   │   └── main.py           # API routes
│   │
│   └── core/                 # Apex core engine
│       ├── __init__.py
│       ├── engine.py         # Main engine logic
│       ├── protocols.py      # Tier and learning protocols
│       ├── risk_filters.py   # Risk management
│       ├── failsafes.py      # Omega directives
│       ├── learning.py       # Learning protocol implementation
│       ├── quantum.py        # Advanced analysis modules
│       └── zde.py            # Zero-drift engine
│
├── cli/                      # Command-line interface
│   └── main.py               # Typer CLI application
│
├── tests/                    # Test suite
│   ├── test_api.py
│   ├── test_cli.py
│   ├── test_engine.py
│   ├── test_protocols.py
│   ├── test_risk_filters.py
│   └── test_zde.py
│
├── docs/                     # Documentation
│   ├── OVERVIEW_QUANTRACORE_APEX.md
│   ├── QUANTRACORE_APEX_MASTER_SPEC_v8_0.yml
│   ├── APEXLAB_OVERVIEW.md
│   ├── APEXCORE_MODELS.md
│   └── ...
│
├── SBOM/                     # Software Bill of Materials
│   ├── sbom.cdx.json
│   ├── provenance.json
│   └── checksums.csv
│
├── assets/                   # Branding and media
│   ├── logo.txt
│   └── screenshots.txt
│
├── scripts/                  # Utility scripts
│   ├── generate_checksums.py
│   └── verify_provenance.py
│
└── archives/                 # Data archives (not in git)
    ├── raw_api_cache/
    └── api_transformed/
```

---

## 4. Getting Started

### 4.1 Prerequisites

- Python 3.10+
- pip or uv package manager
- Git

### 4.2 Installation

```bash
# Clone the repository
git clone https://github.com/Lamont-Labs/QuantraCore.git
cd QuantraCore

# Install dependencies
pip install -r requirements.txt
```

### 4.3 Verify Installation

```bash
# Run tests
pytest -q

# Run determinism verification
bash verify.sh
```

---

## 5. Running Tests

### 5.1 Full Test Suite

```bash
pytest
```

### 5.2 Specific Test Files

```bash
pytest tests/test_engine.py
pytest tests/test_protocols.py
```

### 5.3 Verbose Output

```bash
pytest -v
```

### 5.4 Test Coverage

```bash
pytest --cov=src
```

---

## 6. Running Apex in Research/Simulation Mode

### 6.1 CLI Demo

The CLI provides a quick way to run the Apex engine:

```bash
python -m cli.main
```

This runs the deterministic demo and outputs results to `dist/golden_demo_outputs/`.

### 6.2 API Server

Start the FastAPI server for integration testing:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 5000
```

Available endpoints:
- `GET /health` — Health check
- `GET /score` — QuantraScore computation
- `GET /risk/hud` — Risk HUD data
- `GET /audit/export` — Audit log export

### 6.3 Simulation Mode

The system runs in research/simulation mode by default. No live trading is possible without explicitly enabling execution envelopes (which requires code changes and approvals).

---

## 7. ApexLab Scripts

ApexLab training scripts are located in the `intelligence/apexlab/` directory (when present). These scripts handle:

- Data ingestion and caching
- Feature extraction
- Model training
- Distillation
- Validation
- Export

**Note:** ApexLab requires significant computational resources and is typically run on dedicated hardware (K6). For development purposes, you can review the scripts without running full training.

---

## 8. Style and Naming Conventions

### 8.1 Python Style

- Follow PEP 8
- Use type hints for function signatures
- Docstrings for public functions and classes
- Maximum line length: 100 characters

### 8.2 Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Modules | snake_case | `risk_filters.py` |
| Classes | PascalCase | `ApexEngine` |
| Functions | snake_case | `compute_quantrascore` |
| Constants | UPPER_SNAKE | `MAX_ENTROPY_BAND` |
| Variables | snake_case | `current_regime` |

### 8.3 File Organization

- One primary class per module
- Related utilities in the same module
- Tests mirror source structure

### 8.4 Documentation

- All public APIs documented
- Complex logic explained in comments
- README in each major directory

---

## 9. Key Concepts

### 9.1 Determinism

All core logic must be deterministic:
- No random operations without seeding
- No current-time dependencies in logic
- No floating-point instability
- Reproducible outputs for identical inputs

### 9.2 Proof Logging

All significant operations must be logged:
- Input/output hashes
- Timestamps
- Parameters
- Status

### 9.3 Fail-Closed

When uncertain, restrict rather than guess:
- Validation failures halt processing
- Model uncertainty triggers abstention
- Missing data causes explicit errors

### 9.4 Omega Directives

Understand the four Omega directives:
- **Ω1** — Integrity Lock
- **Ω2** — Risk Kill Switch
- **Ω3** — Config Guard
- **Ω4** — Compliance Gate

---

## 10. Contributing

### 10.1 Branch Strategy

- `main` — Stable, production-ready
- `develop` — Integration branch
- `feature/*` — New features
- `fix/*` — Bug fixes

### 10.2 Pull Request Process

1. Create feature/fix branch
2. Implement changes
3. Add/update tests
4. Ensure all tests pass
5. Update documentation
6. Submit PR for review

### 10.3 Code Review

All changes require:
- At least one approving review
- Passing CI checks
- Documentation updates (if applicable)

---

## 11. Resources

- [System Overview](OVERVIEW_QUANTRACORE_APEX.md)
- [Master Spec v8.0](QUANTRACORE_APEX_MASTER_SPEC_v8_0.yml)
- [API Reference](API_REFERENCE.md)
- [Compliance Policy](COMPLIANCE_POLICY.md)
- [Security Guide](SECURITY_AND_HARDENING.md)

---

## 12. Getting Help

- Review existing documentation
- Check GitHub issues for similar questions
- Contact the development team
- Refer to the master spec for authoritative definitions
