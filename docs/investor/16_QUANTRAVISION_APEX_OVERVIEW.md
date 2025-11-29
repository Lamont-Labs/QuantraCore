# QuantraVision Apex Overview

**Document Classification:** Investor Due Diligence — Technical  
**Version:** 9.0-A  
**Date:** November 2025  

---

## What Is QuantraVision Apex?

QuantraVision Apex is the Android mobile companion app that brings structural analysis to retail investors. It uses on-device neural models to provide visual overlays on live charts, without sending user data to external servers.

---

## Relationship to QuantraCore

| Aspect | QuantraCore Apex | QuantraVision Apex |
|--------|------------------|-------------------|
| **Platform** | Desktop/Server | Android Mobile |
| **Target User** | Institutional | Retail/Prosumer |
| **Codebase** | This repository | Separate repository |
| **Models** | ApexCore v2 Big | ApexCore v2 Mini + VisionLite |
| **Framing** | Research tool | Educational copilot |

**Shared Components:**
- ApexCore v2 Mini architecture
- Core structural concepts
- Safety philosophy

---

## QuantraVision v4.x Pipeline

### Visual Pattern Detection

```
Camera/Screenshot → VisionLite → Bounding boxes
```

**VisionLite** detects:
- Chart boundaries
- Candlestick patterns
- Support/resistance zones
- Trend lines

### Candle Analysis

```
Bounding boxes → CandleLite → Candle data extraction
```

**CandleLite** extracts:
- OHLC values from visual
- Volume patterns
- Time axis parsing

### Structural Analysis

```
Candle data → ApexLite → Structural signals
```

**ApexLite** (lightweight engine) provides:
- Simplified protocol execution
- Core structural scoring
- Pattern recognition

### Neural Enhancement

```
Structural signals → ApexCore Mini → Enhanced outputs
```

**ApexCore Mini** adds:
- Runner probability
- Quality tier prediction
- Regime classification

### Overlay Rendering

```
All signals → HUD Renderer → Visual overlays
```

**HUD displays:**
- Pattern bounding boxes
- QuantraScore badge
- Trend indicators
- Warning annotations

---

## Safety Guardrails

### Guardrail 1: Structural Overlays Only

The app displays:
- Pattern detection boxes
- Structural scores
- Trend indicators

The app never displays:
- "BUY" or "SELL" signals
- Price targets
- Position sizing

### Guardrail 2: Educational Framing

All outputs are framed as educational:

```
"This pattern shows structural characteristics often seen before
significant moves. This is educational information, not financial
advice. Always do your own research."
```

### Guardrail 3: No Trade Signals

The app cannot:
- Place orders
- Connect to brokers
- Generate actionable signals

### Guardrail 4: Privacy-Preserving

All processing is on-device:
- No chart data sent to servers
- No user tracking
- No cloud inference

---

## User Experience

### Live Chart View

1. User opens app on Android device
2. Points camera at chart (or captures screenshot)
3. VisionLite detects chart boundaries
4. Overlays appear showing:
   - Pattern bounding boxes
   - QuantraScore indicator
   - Trend direction badge

### Research Mode

1. User selects symbol from watchlist
2. App fetches public OHLCV data
3. ApexLite analyzes structure
4. Results displayed with:
   - Quality tier badge
   - Key pattern annotations
   - Educational context

---

## Model Specifications

### VisionLite

| Property | Value |
|----------|-------|
| Purpose | Chart/pattern detection |
| Size | ~5MB |
| Inference | <50ms |
| Framework | TensorFlow Lite |

### CandleLite

| Property | Value |
|----------|-------|
| Purpose | Candle extraction |
| Size | ~3MB |
| Inference | <30ms |
| Framework | TensorFlow Lite |

### ApexLite

| Property | Value |
|----------|-------|
| Purpose | Structural analysis |
| Protocols | Subset of full engine |
| Inference | <100ms |
| Implementation | Native code |

### ApexCore Mini

| Property | Value |
|----------|-------|
| Purpose | Neural enhancement |
| Ensemble | 3 models |
| Size | ~15MB |
| Inference | <80ms |
| AUC | 0.754 |

---

## Why Retail Users Need This

| Problem | QuantraVision Solution |
|---------|----------------------|
| No institutional-grade tools | Brings structural analysis to mobile |
| Subjective pattern reading | Quantified structural scoring |
| Information overload | Focused, visual overlays |
| Black-box complexity | Educational, explainable outputs |

---

## Monetization Potential

| Model | Description |
|-------|-------------|
| **Freemium** | Basic overlays free, premium features paid |
| **Subscription** | Monthly access to full feature set |
| **In-App Purchase** | Additional model packs or features |
| **White-Label** | License to broker/platform apps |

---

## Current Status

| Component | Status |
|-----------|--------|
| VisionLite | Operational |
| CandleLite | Operational |
| ApexLite | Operational |
| ApexCore Mini | Operational |
| Android App | v4.x available |
| iOS Version | Not yet developed |

---

## Relationship Summary

```
┌─────────────────────────────────────────────────────┐
│                  SHARED ARCHITECTURE                │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ApexCore v2 Architecture                           │
│  ├── Big (Desktop)  → QuantraCore Apex             │
│  └── Mini (Mobile)  → QuantraVision Apex           │
│                                                      │
│  Structural Concepts                                │
│  ├── QuantraScore                                   │
│  ├── Quality Tiers                                  │
│  └── Protocol Logic                                 │
│                                                      │
│  Safety Philosophy                                  │
│  ├── No trade signals                               │
│  ├── Educational framing                            │
│  └── Fail-closed behavior                           │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

*QuantraCore Apex v9.0-A | Lamont Labs | November 2025*
