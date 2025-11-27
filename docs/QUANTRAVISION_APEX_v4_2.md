# QuantraVision Integration — Mobile Visual Intelligence

**Version:** 8.0  
**Component:** QuantraVision  
**Role:** Mobile overlay copilot for chart analysis

---

## 1. Overview

QuantraVision provides mobile visual intelligence within the QuantraCore Apex ecosystem. It delivers real-time structural analysis of trading charts directly on mobile devices through visual overlays.

The system exists in two versions:
- **v2_apex** — Current version with full ApexCore integration
- **v1_legacy** — Legacy version for basic signal viewing

---

## 2. Version Comparison

| Feature | v2_apex (Current) | v1_legacy |
|---------|-------------------|-----------|
| On-device scan | Yes | No |
| ApexLite | Yes | No |
| ApexCore Mini | Yes | No |
| HUD | Yes | Basic |
| Overlays | Full | Limited |
| Cloud narration | Optional | No |
| Target audience | Pro/Enterprise | Retail |

---

## 3. v2_apex (Current Version)

### 3.1 Features

- **On-device scan** — Local chart analysis without network
- **ApexLite** — Lightweight Apex processing
- **ApexCore Mini** — Full neural model integration
- **HUD** — Heads-up display overlays
- **Overlays** — Visual chart annotations
- **Cloud narration optional** — Text commentary when enabled

### 3.2 Processing Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    QUANTRAVISION v2_apex PIPELINE                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. CAPTURE        Screen capture of trading chart                       │
│        ↓                                                                 │
│  2. BBOX           Bounding box detection to isolate chart region        │
│        ↓                                                                 │
│  3. VISIONLITE     Chart parsing and structure extraction                │
│        ↓                                                                 │
│  4. CANDLELITE     Candle/bar extraction and normalization               │
│        ↓                                                                 │
│  5. PRIMITIVES     Visual primitive detection (support, resistance, etc) │
│        ↓                                                                 │
│  6. APEXLITE       Structural scoring using ApexCore Mini                │
│        ↓                                                                 │
│  7. QUANTRASCORE   Composite quality score (0–100)                       │
│        ↓                                                                 │
│  8. HUD OVERLAY    Visual overlay rendering on screen                    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.3 ApexCore Mini Integration

When ApexCore Mini is available:

```
CandleLite Output → Feature Extraction → ApexCore Mini → Structural Scores
```

**Outputs used from ApexCore Mini:**
- Regime classification
- Risk tier
- Chart quality score
- QuantraScore (0–100)
- Volatility banding
- Entropy states
- Suppression detection

---

## 4. v1_legacy (Legacy Version)

### 4.1 Features

- **Signal viewer only** — Displays upstream signals
- **Thin client for retail** — Minimal local processing
- **Upstream Apex signals** — Receives analysis from server

### 4.2 Use Case

v1_legacy serves as a lightweight viewer for users who:
- Don't need on-device processing
- Have reliable network connectivity
- Prefer server-side analysis
- Are in the retail tier

---

## 5. Safety Constraints

Both versions operate under strict safety rules:

### 5.1 No Trading

The system has **zero execution capability**:
- No order placement
- No account integration
- No broker connections
- No portfolio modifications

### 5.2 No Recommendations

All output is strictly informational:
- No "buy" or "sell" signals
- No price targets
- No financial advice
- Educational content only

### 5.3 Forbidden Phrases

Generated text is validated against forbidden patterns:

```yaml
forbidden_patterns:
  - "buy now"
  - "sell now"
  - "guaranteed"
  - "profit"
  - "you should"
  - "financial advice"
  - "recommendation"
```

---

## 6. HUD Overlay Elements

### 6.1 QuantraScore Badge

Displays the QuantraScore (0–100) with color coding:

| Score Range | Color | Meaning |
|-------------|-------|---------|
| 70–100 | Green | Strong pass |
| 50–69 | Yellow | Pass/Wait |
| 0–49 | Red | Fail/Caution |

### 6.2 Regime Indicator

Shows detected market regime:
- Trending (directional arrow)
- Ranging (horizontal bars)
- Volatile (lightning bolt)
- Suppressed (pause icon)

### 6.3 Risk Level

Displays risk tier with visual indicators:
- Low: Green shield
- Medium: Yellow triangle
- High: Orange warning
- Extreme: Red stop sign

---

## 7. Target Platform

| Platform | Support |
|----------|---------|
| Android | Primary target |
| iOS | Planned |

**Hardware Requirements:**
- Android device with camera/screen capture
- Sufficient processing power for ApexCore Mini (<30ms inference)

---

## 8. Performance Targets

| Metric | v2_apex | v1_legacy |
|--------|---------|-----------|
| End-to-end latency | <500ms | <200ms |
| ApexCore Mini inference | <30ms | N/A |
| Battery impact | <5%/hr | <2%/hr |
| Memory usage | <150MB | <50MB |

---

## 9. Summary

QuantraVision provides mobile visual intelligence through two versions: v2_apex with full ApexCore integration for professional users, and v1_legacy as a lightweight signal viewer for retail users. Both versions maintain strict safety constraints—no trading, no recommendations—while delivering valuable structural analysis through intuitive HUD overlays.
