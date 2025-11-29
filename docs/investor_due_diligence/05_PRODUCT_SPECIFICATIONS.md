# Product Specifications Document

**Document Classification:** Investor Due Diligence  
**Version:** 9.0-A  
**Date:** November 2025  
**Status:** Production  
**Owner:** Lamont Labs

---

## Executive Summary

QuantraCore Apex v9.0-A is a desktop-based institutional-grade deterministic AI trading intelligence engine designed for research and backtesting. The system provides structural market analysis through a proprietary protocol system, neural assistant models, and comprehensive risk management.

---

## Product Overview

### Core Value Proposition

| Aspect | Description |
|--------|-------------|
| **What** | Deterministic AI trading intelligence engine |
| **For** | Institutional researchers, quantitative analysts, trading desks |
| **Problem Solved** | Reliable, reproducible market structure analysis with institutional safety |
| **Differentiator** | Fail-closed design, 3-5x regulatory margins, offline operation |

### Product Category

- **Primary:** Quantitative research platform
- **Secondary:** Risk analysis system
- **Mode:** Research/backtest only (no live trading)

---

## Feature Matrix

### Core Analysis Features

| Feature | Status | Description |
|---------|--------|-------------|
| QuantraScore | ✓ Active | 0-100 composite structural score |
| Regime Detection | ✓ Active | Market regime classification |
| Risk Tier Assessment | ✓ Active | Multi-level risk categorization |
| Entropy Analysis | ✓ Active | Market disorder measurement |
| Drift Detection | ✓ Active | Trend drift monitoring |
| Suppression Detection | ✓ Active | Signal dampening identification |
| Continuation Estimation | ✓ Active | Trend continuation probability |
| MonsterRunner | ✓ Active | Rare event precursor detection |

### Protocol System (115 Protocols)

| Protocol Type | Count | Purpose |
|---------------|-------|---------|
| Tier Protocols (T01-T80) | 80 | Structural analysis rules |
| Learning Protocols (LP01-LP25) | 25 | Training label generation |
| MonsterRunner (MR01-MR05) | 5 | Rare event detection |
| Omega Directives (Ω1-Ω5) | 5 | Safety overrides |

### Neural Assistant Models

| Model | Type | Size | Latency | Heads |
|-------|------|------|---------|-------|
| ApexCore V2 Big | Desktop | 3-20MB | <20ms | 5 |
| ApexCore V2 Mini | Lightweight | 0.5-3MB | <30ms | 5 |
| ApexCore V1 Full | Legacy | 3-20MB | <20ms | 8-12 |
| ApexCore V1 Mini | Legacy | 0.5-3MB | <30ms | Distilled |

### Prediction Heads (ApexCore V2)

| Head | Output | Range |
|------|--------|-------|
| quantra_score | Predicted QuantraScore | 0-100 |
| runner_prob | Monster runner probability | 0-1 |
| quality_tier | Quality classification | A/B/C/D |
| avoid_trade | Trade avoidance signal | 0-1 |
| regime | Predicted market regime | Enum |

---

## Risk Management Features

### Safety Systems

| System | Description | Status |
|--------|-------------|--------|
| Omega Ω1 | Hard safety lock on extreme risk | Standby |
| Omega Ω2 | Entropy-based override | Standby |
| Omega Ω3 | Drift-based override | Standby |
| Omega Ω4 | Compliance mode (research-only) | **Always Active** |
| Omega Ω5 | Signal suppression lock | Standby |

### Risk Assessment Capabilities

| Capability | Description |
|------------|-------------|
| Volatility Gates | Multi-threshold volatility checks |
| Regime Gates | Market regime validation |
| Entropy Gates | Disorder level validation |
| Drift Gates | Trend stability checks |
| Kill Switches | Automatic safety halts |

---

## Data Integration

### Supported Data Providers

| Provider | Type | Status |
|----------|------|--------|
| Polygon.io | Primary market data | Supported |
| Alpha Vantage | Alternative source | Supported |
| Yahoo Finance | Backup provider | Supported |
| CSV Bundle | Historical import | Supported |
| Synthetic | Testing/demo | Supported |

### Data Types Supported

| Type | Format | Usage |
|------|--------|-------|
| OHLCV | Daily/Intraday | Primary analysis |
| Volume | Bar-by-bar | Volume analysis |
| Fundamentals | Quarterly | Context (optional) |

---

## User Interface

### ApexDesk Dashboard

| Component | Description |
|-----------|-------------|
| Signal Grid | Multi-symbol QuantraScore display |
| Protocol Explorer | Protocol execution visualization |
| Entropy Console | Real-time entropy monitoring |
| Drift Display | Trend drift indicators |
| Risk HUD | Heads-up risk display |
| Predictive Panel | V2 prediction display |
| Compliance Status | Regulatory score display |

### Visualization Features

| Feature | Description |
|---------|-------------|
| Real-time Updates | Live analysis refresh |
| Symbol Comparison | Side-by-side analysis |
| Historical View | Past analysis review |
| Export Capability | JSON/CSV export |

---

## API Capabilities

### Endpoint Summary

| Category | Endpoints | Purpose |
|----------|-----------|---------|
| Analysis | 8 | Core scanning and tracing |
| Risk | 3 | Risk assessment and signals |
| Compliance | 5 | Regulatory monitoring |
| Predictive | 4 | Advisory layer access |
| Engine/System | 7 | System status and utilities |
| Portfolio | 9 | Simulation management |
| **Total** | **36** | Complete API coverage |

### Integration Features

| Feature | Description |
|---------|-------------|
| REST API | Standard HTTP/JSON interface |
| OpenAPI Docs | Auto-generated documentation |
| SDK Support | Python/JavaScript examples |
| Webhooks | Event notification (planned) |

---

## Deployment Specifications

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8 cores |
| RAM | 8 GB | 16 GB |
| Storage | 10 GB SSD | 50 GB SSD |
| OS | Linux/macOS/Windows | Linux |
| Python | 3.11 | 3.11 |
| Node.js | 18+ | 20+ |

### Deployment Options

| Option | Description | Use Case |
|--------|-------------|----------|
| Desktop | Local installation | Individual researcher |
| Server | On-premise deployment | Team access |
| Air-gapped | Network-isolated | High security |

---

## Product Roadmap

### Current Release (v9.0-A)

- ✓ Institutional hardening complete
- ✓ Predictive layer V2 implemented
- ✓ Regulatory excellence framework
- ✓ 970 tests passing
- ✓ 99.25% compliance score

### Near-Term (v9.1)

| Feature | Priority | Timeline |
|---------|----------|----------|
| Model training pipeline | High | Q1 2026 |
| Enhanced visualization | Medium | Q1 2026 |
| Additional data providers | Medium | Q1 2026 |
| Performance optimization | Medium | Q1 2026 |

### Medium-Term (v10.0)

| Feature | Priority | Timeline |
|---------|----------|----------|
| Multi-asset support | High | Q2 2026 |
| Advanced backtesting | High | Q2 2026 |
| Team collaboration | Medium | Q2 2026 |
| Enterprise features | Medium | Q3 2026 |

---

## Competitive Positioning

### Unique Differentiators

| Differentiator | Description |
|----------------|-------------|
| **Determinism** | 100% reproducible outputs |
| **Fail-Closed** | Always defaults to safe state |
| **Offline-First** | No cloud dependencies |
| **Regulatory Excellence** | 3-5x stricter than required |
| **Protocol System** | 115 proprietary analysis protocols |
| **Comprehensive Testing** | 970 automated tests |

### Target Market Segments

| Segment | Use Case |
|---------|----------|
| Hedge Funds | Research validation |
| Prop Trading | Signal analysis |
| Asset Managers | Risk assessment |
| Quantitative Researchers | Academic/commercial research |
| Family Offices | Investment analysis |

---

## Compliance Constraints

### Architectural Constraints

| Constraint | Implementation |
|------------|----------------|
| No live trading | OMS disabled by default |
| Research-only mode | Omega Ω4 always active |
| No trade recommendations | Structural analysis only |
| Desktop-only | No mobile builds |

### Regulatory Positioning

| Statement | Description |
|-----------|-------------|
| Purpose | Research and backtesting platform |
| Outputs | Structural probabilities, not advice |
| Liability | Users responsible for decisions |
| Compliance | Exceeds SEC/FINRA/MiFID II/Basel |

---

## Support & Documentation

### Documentation Library

| Category | Documents |
|----------|-----------|
| Technical | 25+ files |
| API Reference | Complete |
| Architecture | Comprehensive |
| Protocols | Full inventory |
| Security | Detailed |
| Compliance | Extensive |

### Support Tiers (Planned)

| Tier | Description |
|------|-------------|
| Community | Documentation, forums |
| Standard | Email support, updates |
| Enterprise | Dedicated support, customization |

---

*Document prepared for investor due diligence. Confidential.*
