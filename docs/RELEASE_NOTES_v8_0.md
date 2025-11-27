# Release Notes — QuantraCore Apex v8.0

**Release Date:** October 2025  
**Version:** 8.0  
**Codename:** Apex

---

## Overview

QuantraCore Apex v8.0 represents a major evolution of the QuantraCore platform, consolidating previous versions into a unified, acquisition-ready architecture. This release introduces comprehensive API integration specifications, refined model family definitions, and enhanced compliance controls.

---

## What's New in v8.0

### 1. Master Spec Consolidation

Version 8.0 introduces the **Master System Specification**, a single YAML document that serves as the canonical source of truth for all system components, interfaces, and behaviors.

**Benefits:**
- Single reference for all stakeholders
- Machine-readable for automated validation
- Complete ecosystem definition
- Clear version control

**Location:** `docs/QUANTRACORE_APEX_MASTER_SPEC_v8_0.yml`

---

### 2. API Integration Specification

New in v8.0: a complete specification for external API integration, covering:

- **Deterministic contract** for all adapters
- **Caching and replay** requirements
- **Supported data sources:**
  - Market data (IEX, Polygon, Alpaca)
  - Fundamentals (financials, earnings)
  - Volatility (options, VIX)
  - Macro (economic indicators, sector indices)
  - Alternatives (short interest, insider trades)
  - News (text-only feeds)

**Key Principles:**
- All raw API responses cached before processing
- Deterministic transformations only
- Hash logging for all operations
- Fail-closed on schema drift

**Documentation:** `docs/API_INTEGRATION_FOR_LEARNING_AND_PREDICTION.md`

---

### 3. ApexCore Model Family Refinement

Version 8.0 clarifies the ApexCore model architecture:

**ApexCore Full:**
- Role: Desktop structural model
- Size: 4–20MB
- Training: Direct from ApexLab

**ApexCore Mini:**
- Role: Mobile structural model
- Size: 0.5–3MB
- Training: Distilled from Full
- Inference: <30ms on mobile

**Shared Output Contract:**
- Regime, risk tier, chart quality
- Entropy band, volatility band
- Suppression state, score band
- QuantraScore (numeric)

**Documentation:** `docs/APEXCORE_MODELS.md`

---

### 4. Broker/OMS Execution Envelope

Version 8.0 introduces explicit **execution envelope** controls:

**Default State:**
```yaml
broker_layer:
  default:
    execution_enabled: false
```

**Available Modes:**
- `disabled` — No execution path (default)
- `simulation` — Paper trading with fake orders
- `paper` — Paper trading with broker connection
- `live_ready` — Production execution (requires approval)

**Key Control:**
- Execution is disabled by default
- Mode transitions require Ω4 (Compliance Gate) approval
- All transitions are audit-logged
- Research-first design satisfies regulatory classification

---

### 5. Enhanced Documentation Suite

Version 8.0 includes a comprehensive documentation refresh:

| Document | Description |
|----------|-------------|
| Overview | Full narrative of the Apex ecosystem |
| Master Spec | Canonical YAML specification |
| ApexLab Overview | Offline training lab documentation |
| ApexCore Models | Model family specifications |
| QuantraVision Apex | Mobile overlay copilot (v4.2) |
| QuantraVision Remote | Desktop-to-mobile streaming |
| Prediction & MonsterRunner | Forward-looking analysis |
| API Integration | External data adapter specs |
| Compliance Policy | Institutional compliance guide |
| Security & Hardening | System hardening documentation |
| Developer Guide | Engineer onboarding |
| Release Notes | This document |

---

## Changes from v7.0 and Earlier

### Architectural Changes

| Area | v7.0 and Earlier | v8.0 |
|------|------------------|------|
| Specification | Multiple documents | Single master spec |
| Model naming | Various names | ApexCore Full/Mini |
| API integration | Ad-hoc | Formal specification |
| Execution | Implicit controls | Explicit envelopes |

### Terminology Updates

| Old Term | New Term |
|----------|----------|
| NeuroCore | QuantraCore Apex |
| Core models | ApexCore Full/Mini |
| Training system | ApexLab |
| Mobile vision | QuantraVision Apex v4.2 |

### Removed Features

- Legacy model formats (replaced by TFLite standard)
- Deprecated API adapters (consolidated to supported list)
- Old configuration schemas (migrated to v8.0 format)

---

## Migration Guide

### From v7.x to v8.0

1. **Update configuration files** to v8.0 schema
2. **Replace legacy model references** with ApexCore Full/Mini
3. **Update API adapter configurations** to new format
4. **Review execution envelope settings** (now explicit)
5. **Update documentation references** to new paths

### Breaking Changes

- Configuration schema changes (migration required)
- Model format standardization (re-export may be needed)
- API adapter interface updates (adapter code changes)

---

## Known Limitations

- ApexLab training requires dedicated K6 hardware
- QuantraVision Remote requires stable network connection
- Mobile model size constraints limit some accuracy

---

## Future Roadmap

- v8.1: Enhanced MonsterRunner feature set
- v8.2: Additional API adapter support
- v9.0: Next-generation model architectures

---

## Contributors

- Jesse J. Lamont — Founder, Lamont Labs
- Lamont Labs Engineering Team

---

## Acknowledgments

Thank you to all reviewers, testers, and stakeholders who contributed to the development and refinement of QuantraCore Apex v8.0.

---

**QuantraCore Apex v8.0**  
*Deterministic. Reproducible. Acquisition-Ready.*
