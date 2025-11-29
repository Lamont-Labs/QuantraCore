# Engineering Overview and Practices

**Document Classification:** Investor Due Diligence — Engineering  
**Version:** 9.0-A  
**Date:** November 2025  

---

## How Is the Code Structured and Maintained?

This document provides an overview of the codebase structure, coding standards, and engineering practices.

---

## Directory Layout

```
quantracore-apex/
├── src/
│   └── quantracore_apex/
│       ├── engine/           # Deterministic analysis engine
│       ├── protocols/        # Protocol implementations
│       │   ├── tier/         # T01-T80 Tier protocols
│       │   ├── learning/     # LP01-LP25 Learning protocols
│       │   ├── monster_runner/ # MR01-MR05 protocols
│       │   └── omega/        # Ω1-Ω5 directives
│       ├── apexlab/          # Labeling and dataset generation
│       ├── apexcore/         # Neural model inference
│       ├── advisor/          # PredictiveAdvisor
│       ├── scanner/          # Market scanning
│       ├── feed/             # Data ingestion
│       ├── risk/             # Risk management
│       ├── server/           # FastAPI backend
│       └── utils/            # Shared utilities
├── dashboard/                # React frontend (ApexDesk)
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── pages/            # Page components
│   │   ├── hooks/            # Custom hooks
│   │   ├── services/         # API services
│   │   └── utils/            # Frontend utilities
│   └── public/               # Static assets
├── tests/                    # Test suites
│   ├── core/                 # Core engine tests
│   ├── protocols/            # Protocol tests
│   ├── apexlab/              # Lab tests
│   ├── apexcore/             # Model tests
│   ├── nuclear/              # Safety-critical tests
│   ├── extreme/              # Edge case tests
│   └── integration/          # Integration tests
├── models/                   # Model artifacts
│   └── apexcore_v2/
│       ├── big/              # Desktop model
│       └── mini/             # Mobile model
├── config/                   # Configuration files
├── docs/                     # Documentation
│   ├── investor/             # Investor documentation
│   └── assets/               # Diagrams and images
├── scripts/                  # Utility scripts
├── metrics/                  # Performance metrics
└── proof_logs/               # Audit logs
```

---

## Coding Standards

### Python Style

| Standard | Implementation |
|----------|----------------|
| Formatter | Black (line length 88) |
| Linter | Ruff (fast, comprehensive) |
| Type hints | Required for public APIs |
| Docstrings | Google style |
| Naming | snake_case for functions, PascalCase for classes |

### TypeScript Style

| Standard | Implementation |
|----------|----------------|
| Formatter | Prettier |
| Linter | ESLint |
| Types | Strict mode enabled |
| Components | Functional with hooks |
| Naming | camelCase for functions, PascalCase for components |

### Example Python Code

```python
from typing import Optional
from dataclasses import dataclass

@dataclass
class ScanResult:
    """Result of a single symbol scan.
    
    Attributes:
        symbol: The ticker symbol analyzed.
        quantra_score: Structural score (0-100).
        quality_tier: Quality classification.
        timestamp: Analysis timestamp.
    """
    symbol: str
    quantra_score: float
    quality_tier: str
    timestamp: str
    
    def is_high_quality(self) -> bool:
        """Check if result meets high quality threshold."""
        return self.quality_tier in ("A+", "A")
```

### Example TypeScript Code

```typescript
interface ScanResult {
  symbol: string;
  quantraScore: number;
  qualityTier: string;
  timestamp: string;
}

const ScanResultCard: React.FC<{ result: ScanResult }> = ({ result }) => {
  const isHighQuality = ['A+', 'A'].includes(result.qualityTier);
  
  return (
    <div className={`card ${isHighQuality ? 'high-quality' : ''}`}>
      <h3>{result.symbol}</h3>
      <p>Score: {result.quantraScore}</p>
    </div>
  );
};
```

---

## Dependency Philosophy

### Principles

1. **CPU-Friendly:** No GPU requirements for inference
2. **Deterministic:** Reproducible outputs across runs
3. **Minimal:** Fewer dependencies = smaller attack surface
4. **Pinned:** All versions locked for reproducibility

### Core Dependencies

| Package | Purpose | Why This One |
|---------|---------|--------------|
| NumPy | Numerical operations | Standard, fast, stable |
| Pandas | Data manipulation | Standard for financial data |
| scikit-learn | ML models | CPU-only, deterministic, small |
| FastAPI | Web framework | Modern, async, fast |
| React | Frontend framework | Industry standard, stable |
| Vite | Build tool | Fast, modern, simple |

### What We Don't Use

| Package | Reason |
|---------|--------|
| PyTorch/TensorFlow | Too heavy, GPU dependency |
| Django | Overkill for our needs |
| jQuery | Outdated for React apps |
| Large ML frameworks | Disk space, complexity |

---

## Testing Philosophy

### Test Pyramid

```
         /\
        /  \  Integration (few, expensive)
       /----\
      /      \  Unit (many, cheap)
     /--------\
```

### Test Categories

| Category | Purpose | Count |
|----------|---------|-------|
| Core | Engine logic | ~200 |
| Protocols | Protocol correctness | ~300 |
| ApexLab | Labeling logic | ~100 |
| ApexCore | Model inference | ~100 |
| Nuclear | Safety-critical | ~100 |
| Extreme | Edge cases | ~50 |
| Integration | End-to-end | ~50 |

### Determinism Testing

Special tests verify deterministic behavior:

```python
def test_engine_determinism():
    """Same inputs must produce same outputs."""
    result_1 = engine.analyze(test_data)
    result_2 = engine.analyze(test_data)
    
    assert result_1.output_hash == result_2.output_hash
    
    # Run 150 times (3x FINRA requirement)
    for _ in range(150):
        result_n = engine.analyze(test_data)
        assert result_n.output_hash == result_1.output_hash
```

---

## Code Quality Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Test coverage | >80% | ~85% |
| Type coverage | >90% | ~92% |
| Linting errors | 0 | 0 |
| Documentation | Complete | Complete |

---

## Development Workflow

### Local Development

```bash
# Backend
cd src
python -m pytest tests/
uvicorn quantracore_apex.server.app:app --reload

# Frontend
cd dashboard
npm run dev
npm test
```

### Code Review

| Requirement | Description |
|-------------|-------------|
| Tests pass | All tests must pass |
| Linting | No linting errors |
| Types | Type check passes |
| Review | At least one reviewer |

### Deployment

```bash
# Build
npm run build  # Frontend
pytest         # Backend tests

# Deploy (when production-ready)
# Would use CI/CD pipeline
```

---

## Technical Debt Tracking

| Area | Debt | Priority | Plan |
|------|------|----------|------|
| Test coverage | Some edge cases untested | Medium | Continuous improvement |
| Error messages | Some generic | Low | Refine as found |
| Performance | Some unoptimized paths | Low | Profile when needed |
| Documentation | Some internal docs stale | Low | Update during development |

---

*QuantraCore Apex v9.0-A | Lamont Labs | November 2025*
