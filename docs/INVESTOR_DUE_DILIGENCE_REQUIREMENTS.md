# Investor & Acquirer Due Diligence Requirements

**Purpose:** Complete requirements checklist for investor, VC, accelerator, PE firm, or acquirer due diligence  
**Author:** Jesse J. Lamont — Lamont Labs  
**Status:** Requirements Specification

---

## Overview

This document defines everything a serious investor or acquirer expects to review before funding or acquiring QuantraCore Apex™.

### Categories

1. Technical Infrastructure
2. Documentation & Architecture
3. Model, Data, Training Artifacts
4. Product & UI Proof
5. Compliance & Safety
6. Business & Ownership
7. Risk, Security, Legal
8. Founder Package

---

## Table of Contents

1. [Technical Infrastructure](#1-technical-infrastructure)
2. [Documentation & Architecture](#2-documentation--architecture)
3. [Model, Data, Training Artifacts](#3-model-data-training-artifacts)
4. [Product & UI Proof](#4-product--ui-proof)
5. [Compliance & Safety](#5-compliance--safety)
6. [Business, Market & Strategy](#6-business-market--strategy)
7. [Risk, Security & Legal](#7-risk-security--legal)
8. [Founder Package](#8-founder-package)
9. [Optional High-Value Items](#9-optional-high-value-items)

---

## 1. Technical Infrastructure

### Must Include

| Item | Description |
|------|-------------|
| README | Executive summary + architecture overview + quickstart |
| Architecture Diagrams | Full system diagrams (SVG + PNG) |
| Directory Map | Folder structure with explanations |
| Bootstrap Scripts | One-command setup scripts |
| Build Instructions | Reproducible builds (Docker + Makefile) |
| Environment Files | `.env.example`, secrets guidance |
| Dependencies | Pinned versions (requirements.txt + lockfiles) |
| Replit Scripts | apex_auto_debug, apex_superrun, chunked tests |
| Test Suite | 500–700+ tests (unit + scenario + nuclear) |
| CI/CD Pipeline | GitHub Actions + test matrix |
| API Documentation | OpenAPI/Swagger or JSON schema |
| UI Build Instructions | ApexDesk build and run |
| Android Instructions | QuantraVision Apex build + emulator |

### Evidence of Stability

| Evidence | Description |
|----------|-------------|
| Test Status | All tests green |
| Coverage Badge | Test coverage percentage |
| Type Checking | mypy or pyright mandatory |
| Linting | ruff + black mandatory |
| Determinism Proof | Same input → same output logs |
| Performance Benchmarks | Per-symbol latency, scanner benchmarks |
| Resource Profiling | Memory and CPU profiling logs |

---

## 2. Documentation & Architecture

### System Documents

| Document | Status |
|----------|--------|
| Master Specification (v9.x) | PDF + Markdown |
| Engine Architecture Whitepaper | Required |
| Protocol Index (T01–T80, LP01–LP25, MRxx) | Required |
| Omega Safety Framework | Required |
| Zero-Heat (ZDE) Risk Framework | Required |
| ApexFeed Data Ingestion Architecture | Required |
| ApexDesk UI Spec | Required |
| QuantraVision Apex Spec | Required |
| Broker Layer Spec | Required |
| Replit Orchestration Spec | Required |
| Fail-Closed Principles | Required |
| Model Manifest Spec (ApexCore v2) | Required |
| ApexLab v2 Label Schema & Methodology | Required |
| Security and Hashing Architecture | Required |

### Required Diagrams

| Diagram | Description |
|---------|-------------|
| Full System Flow | Scanner → Engine → ML → Advisors → UI |
| Protocol Engine Graph | Protocol execution flow |
| ApexLab v2 Data Flow | Training data pipeline |
| ApexCore v2 Model Flow | Big/Mini model architecture |
| PredictiveAdvisor Logic Tree | Decision tree |
| Broker Safety Gating | Safety checks flow |
| CI/CD Pipeline | Build and test pipeline |
| Android Subsystems | Mobile architecture |

### Quick Overview Documents

| Document | Audience |
|----------|----------|
| FAQ for Investors | Investor questions |
| Glossary | All terms defined |
| High-Level Narrative | Non-technical exec version |

---

## 3. Model, Data, Training Artifacts

### Model Artifacts Required

| Artifact | Description |
|----------|-------------|
| Model Manifests | SHA256 hashes, metadata, creation dates |
| ApexCore v2 Big | ONNX/Torch format |
| ApexCore v2 Mini | TFLite/int8 format |
| Sample Outputs | Known symbol predictions |
| Calibration Curves | Probability calibration |
| Runner Plots | runner_prob → realized return |
| Quality Tier Hitrates | Tier prediction accuracy |
| Regime Breakdowns | Performance by regime |
| Disagreement Metrics | Ensemble member variance |

### Dataset Artifacts Required

| Artifact | Description |
|----------|-------------|
| Dataset Manifest | Snapshot ID, time range, sectors |
| Sample Parquet | Non-sensitive subset |
| Feature Summary | Feature distributions |
| Target Distribution | Label distributions |
| Labeling Notebook | Demonstration notebook |
| Walk-Forward Report | Validation methodology |

### Training Documentation

| Document | Contents |
|----------|----------|
| Training Scripts | ApexLab v2 training + evaluation |
| Config Files | Training configurations |
| Hyperparameters | Parameter sets used |
| Loss Curves | Training loss over time |
| Metric Curves | Metrics by epoch |
| Promotion Logs | Best checkpoint selection |

### Determinism Evidence

| Evidence | Proof |
|----------|-------|
| Model Manifest | Same hash on repeated build |
| Dataset Hash | Same hash on rebuild |
| Protocol Outputs | Same outputs on rerun |

---

## 4. Product & UI Proof

### ApexDesk UI

| Item | Description |
|------|-------------|
| Screenshots | All views documented |
| Demo Video | Scanner + engine + predictive integration |
| Signal Viewer | Signal display demonstration |
| Protocol Trace | Trace visualization demo |
| Runner Dashboard | Runner detection demo |
| Model Viewer | Model info display |
| Settings | Theming, responsiveness |

### QuantraVision Apex (Android)

| Item | Description |
|------|-------------|
| Screenshots | On-device captures |
| Overlay Demo | Visual overlay demonstration |
| HUD Demo | Heads-up display |
| BBox Demo | Bounding box detection |
| CandleLite Demo | Candle parsing |
| Fail-Closed Examples | Safety demonstrations |
| Video Demo | Floating icon + charts |

### End-to-End Demo

| Demo | Description |
|------|-------------|
| Full Pipeline | Scan → Engine → ML → UI → Vision |
| Symbol Deep Dive | Trace, score, runner_prob |

---

## 5. Compliance & Safety

### Required Documents

| Document | Description |
|----------|-------------|
| Model Usage Safety Policy | How models should be used |
| PredictiveAdvisor Constraints | Limitation documentation |
| Omega Safety Directives | Safety override rules |
| Compliance Framing | Research-only, structural analysis |
| Risk Disclaimers | User risk acknowledgments |
| Data Usage Compliance | Polygon API compliance |
| No Retail Signals | No financial advice policy |
| Fail-Closed Guarantees | Safety mechanism documentation |
| Audit Trail Spec | Complete audit logging |
| Trade Replay Engine | Replay documentation |
| Integrity Hashing Spec | Hash verification |
| Broker Safety Rules | Paper trading by default |

### Investor-Requested

| Document | Description |
|----------|-------------|
| Regulator-Safe UX | Compliant UI framing |
| Not Auto-Trader | Why system is not retail |
| Legal Memo | Prepared with counsel |

---

## 6. Business, Market & Strategy

### Required Documents

| Document | Description |
|----------|-------------|
| Executive Summary | Founder, vision, market |
| Pitch Deck | PDF, 10–15 slides |
| Market Analysis | Size, competitors, positioning |
| Go-to-Market Plan | Launch strategy |
| Pricing Options | Institutional, licensing, API |
| Roadmap | 12–24 months |
| Risks & Mitigation | Risk management |
| Value Proposition | Core value summary |
| Acquisition Analysis | Exit potential |
| Founder Story | Origin statement |

### Investor-Oriented Narrative

| Topic | Explanation |
|-------|-------------|
| Deterministic + Predictive | Why hybrid matters |
| Defensible Architecture | Why Apex is unique |
| Institutional Without Team | Solo founder capability |
| Exit Paths | Fintech, brokers, hedge funds, data providers |

---

## 7. Risk, Security & Legal

### Security Requirements

| Item | Description |
|------|-------------|
| Security Architecture | Hashing, manifests, sandboxing |
| API Key Handling | Key management policy |
| Permissions Model | Access control |
| Layer Isolation | Predictive vs execution separation |
| Broker Safety Gating | Trade safety checks |
| Data Retention Policy | Data lifecycle |
| Backup & Recovery | Disaster recovery plan |
| Privacy Policy | User data handling |
| Threat Modeling | Security threats analysis |
| Attack Surfaces | Vulnerability listing |
| Dependency Check | Vulnerability scan report |

### Legal Documents

| Document | Description |
|----------|-------------|
| Ownership Statement | 100% founder owned |
| Trademark Usage | ™ explanation |
| Licensing | GPL/MIT/Proprietary hybrid |
| Third-Party Compliance | Model license compliance |
| Non-Advice Disclaimer | Not financial advice |
| Terms & Conditions | Draft terms |

---

## 8. Founder Package

### Documents Required

| Document | Description |
|----------|-------------|
| Founder Profile | Background, strengths, weaknesses |
| Psychological Map | Optional but powerful |
| Vision Statement | Long-term vision |
| Execution Philosophy | How work gets done |
| Progress Tracking | All progress to date |
| Full Timeline | First idea → spec → MVP |
| Persistence Proof | Screenshots, logs, demos |

### Credibility Evidence

| Evidence | Description |
|----------|-------------|
| Master Spec | Deep system architecture |
| AI Engineering | Prompting mastery + execution logs |
| System Outputs | Proven on minimal mobile workflow |
| Extreme Tests | Performance evidence |

---

## 9. Optional High-Value Items

| Item | Value |
|------|-------|
| Synthetic Dataset Pipeline | Data generation capability |
| Model Embeddings Visualization | Model understanding |
| Sector Performance Heatmaps | Performance analysis |
| Advanced Backtests | Research-only backtesting |
| Institutional Scaling Plan | Cluster scalability |
| Docker Images | Ready for EC2/DockerHub |
| Investor Sandbox Mode | Safe limited demo |
| Demo Viewer | Canned data demonstration |
| UI Tour Video | Narrated walkthrough |
| Test Dashboard | Visual test results |
| Performance Benchmarks | vs traditional quant methods |

---

## Checklist Summary

### Critical (Must Have)

- [ ] Technical infrastructure complete
- [ ] All tests passing (500–700+)
- [ ] System documentation complete
- [ ] Model artifacts with manifests
- [ ] Product demos (screenshots + video)
- [ ] Compliance documentation
- [ ] Business pitch materials
- [ ] Security documentation
- [ ] Founder package

### Important (Should Have)

- [ ] Architecture diagrams
- [ ] Training documentation
- [ ] Walk-forward validation
- [ ] End-to-end demo video
- [ ] Legal documents drafted

### Nice to Have

- [ ] Investor sandbox mode
- [ ] Docker images
- [ ] Performance visualizations
- [ ] Narrated UI tour

---

*Investor Due Diligence Requirements | QuantraCore Apex™ | Lamont Labs*
