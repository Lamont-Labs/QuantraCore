# Platform Overview

**Document Classification:** Investor Due Diligence  
**Version:** 9.0-A  
**Date:** November 2025  

---

## How All Components Fit Together

QuantraCore Apex v9.x is a modular system where each component has a specific role in the deterministic-to-neural pipeline.

---

## Component Inventory

### 1. Scanner

**Purpose:** Scans the market universe to identify candidates for analysis.

**Status:** Implemented

**Capabilities:**
- 8 scan modes (momentum, breakout, reversal, volatility, etc.)
- 7 market cap buckets (nano to mega)
- Configurable filters (liquidity, sector, price range)
- Batch processing for full-universe scans

**Code Path:** `src/quantracore_apex/scanner/`

---

### 2. Deterministic Engine (ApexEngine)

**Purpose:** The core analysis engine that applies 115 protocols to generate structural signals.

**Status:** Implemented

**Capabilities:**
- 80 Tier Protocols (T01-T80): Structural analysis rules
- 25 Learning Protocols (LP01-LP25): Label generation for training
- 5 MonsterRunner Protocols (MR01-MR05): Extreme move detection
- 5 Omega Directives (Ω1-Ω5): Safety overrides

**Key Properties:**
- 100% deterministic (same inputs → same outputs)
- Hash-verified outputs for audit
- Full trace logging per decision
- Seed-controlled randomness (none by default)

**Code Path:** `src/quantracore_apex/engine/`, `src/quantracore_apex/protocols/`

---

### 3. ApexLab v2

**Purpose:** Offline labeling environment that generates training datasets from historical data.

**Status:** Implemented

**Capabilities:**
- Window-based feature extraction (100-300 bars)
- Teacher labeling via deterministic engine
- Future outcome calculation (returns, drawdown, runup)
- Quality tier assignment (A+/A/B/C/D)
- Parquet/Arrow dataset export

**Key Properties:**
- Strictly offline (no cloud dependencies)
- Lookahead prevention via time-split
- Configurable label schema (40+ fields)

**Code Path:** `src/quantracore_apex/apexlab/`

---

### 4. ApexCore v2 (Big/Mini)

**Purpose:** Neural models that learn from ApexLab datasets to provide ranking assistance.

**Status:** Implemented

**Variants:**
- **Big:** Desktop/server deployment, 5-model ensemble, AUC 0.782
- **Mini:** Mobile/lightweight deployment, 3-model ensemble, AUC 0.754

**Prediction Heads:**
- `quantra_score`: Predicted structural score (0-100)
- `runner_prob`: Probability of significant move
- `quality_tier`: Predicted quality tier
- `avoid_trade`: Warning probability
- `regime`: Market regime classification

**Key Properties:**
- Manifest verification with SHA256 hashes
- Fail-closed on integrity failure
- Never overrides engine authority
- Disagreement detection across ensemble

**Code Path:** `src/quantracore_apex/apexcore/`

---

### 5. PredictiveAdvisor

**Purpose:** Fail-closed ranker that combines engine outputs with model signals.

**Status:** Implemented

**Capabilities:**
- Takes top-N candidates from deterministic engine
- Applies ApexCore v2 for ranking features
- Enforces safety gates (disagreement, avoid_trade thresholds)
- Outputs ranked list with annotations

**Key Properties:**
- **Ranker-only** — cannot generate candidates, only reorder
- Engine and Omega always override
- No "buy/sell" signals, only structural hints
- Fail-closed on model issues

**Code Path:** `src/quantracore_apex/advisor/`

---

### 6. ApexDesk UI

**Purpose:** React/Vite dashboard for research analysts to interact with the system.

**Status:** Implemented

**Views:**
- Scanner results with filtering
- Signal details with protocol traces
- Proof explorer for audit
- Model metrics and calibration
- Lab status and dataset info

**Technology:** React 18, Vite 5, TypeScript, Tailwind CSS

**Code Path:** `dashboard/`

---

### 7. FastAPI Backend

**Purpose:** REST API exposing engine, scanner, and advisory capabilities.

**Status:** Implemented

**Endpoints:** 36 REST endpoints covering:
- Health and system stats
- Single and batch symbol scanning
- Protocol tracing
- Risk assessment
- Compliance scoring
- Portfolio status

**Code Path:** `src/quantracore_apex/server/`

---

### 8. QuantraVision Apex

**Purpose:** Android mobile app for retail investors with structural overlays.

**Status:** Separate codebase (v4.x operational)

**Capabilities:**
- Visual pattern detection (bounding boxes)
- VisionLite + CandleLite models
- ApexLite + optional ApexCore Mini
- HUD overlays on live charts
- Educational framing (no trade signals)

**Relationship:** Shares ApexCore Mini architecture; separate codebase and brand

---

### 9. Broker Layer

**Purpose:** Optional execution envelope for paper/live trading.

**Status:** Spec-only (disabled by default)

**Design:**
- Paper trading as default route
- Risk module with kill-switch (Ω2)
- Explicit configuration required for live
- Fail-closed on any error

**Note:** Not intended for production use without significant additional development.

---

## Implementation Status Summary

| Component | Status | Code Exists | Tests Exist |
|-----------|--------|-------------|-------------|
| Scanner | Implemented | Yes | Yes |
| Deterministic Engine | Implemented | Yes | Yes |
| ApexLab v2 | Implemented | Yes | Yes |
| ApexCore v2 | Implemented | Yes | Yes |
| PredictiveAdvisor | Implemented | Yes | Yes |
| ApexDesk UI | Implemented | Yes | Yes |
| FastAPI Backend | Implemented | Yes | Yes |
| QuantraVision | Separate | Separate | Separate |
| Broker Layer | Spec-only | Minimal | No |

---

## Architecture Diagram

![QuantraCore Apex Ecosystem](../assets/investor/01_quantracore_apex_ecosystem.png)

---

## Key Specifications

For detailed technical specifications, see:

- [System Architecture Deep Dive](10_SYSTEM_ARCHITECTURE_DEEP_DIVE.md)
- [Engine and Protocols](11_ENGINE_AND_PROTOCOLS.md)
- [ApexLab and ApexCore Models](12_APEXLAB_AND_APEXCORE_MODELS.md)

---

*QuantraCore Apex v9.0-A | Lamont Labs | November 2025*
