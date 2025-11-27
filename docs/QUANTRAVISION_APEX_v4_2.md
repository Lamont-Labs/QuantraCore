# QuantraVision Apex v4.2 — Mobile Overlay Copilot

**Version:** 4.2  
**Component:** QuantraVision Apex (Mobile)  
**Role:** Real-time chart analysis overlay for mobile devices

---

## 1. Overview

QuantraVision Apex v4.2 is the mobile overlay copilot within the QuantraCore Apex ecosystem. It provides real-time structural analysis of trading charts directly on a user's mobile device, presenting insights as unobtrusive visual overlays.

Key characteristics:
- **On-Device Processing** — All analysis runs locally on the mobile device
- **Overlay-Based UX** — Results appear as HUD elements over existing apps
- **Read-Only** — No trading or execution capabilities
- **Safety-First** — Strict content filtering and forbidden phrase validation

---

## 2. Full Pipeline

QuantraVision Apex processes chart images through a multi-stage pipeline:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    QUANTRAVISION APEX v4.2 PIPELINE                      │
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
│  6. APEXLITE       Structural scoring using ApexCore Mini (if available) │
│        ↓                                                                 │
│  7. QUANTRASCORE   Composite quality score computation                   │
│        ↓                                                                 │
│  8. HUD OVERLAY    Visual overlay rendering on screen                    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Stage Details

| Stage | Input | Output | Model/Logic |
|-------|-------|--------|-------------|
| Capture | Screen | Image buffer | System API |
| Bbox | Image | Chart coordinates | BboxModel (TFLite) |
| VisionLite | Cropped chart | Parsed structure | VisionLite model |
| CandleLite | Parsed structure | OHLC data | CandleLite model |
| Primitives | OHLC + structure | Visual primitives | Rule-based + ML |
| ApexLite | Features | Structural scores | ApexCore Mini |
| QuantraScore | All inputs | Composite score | Weighted formula |
| HUD | Scores + primitives | Visual overlay | Rendering engine |

---

## 3. Use of ApexCore Mini

When ApexCore Mini is available on the device, QuantraVision leverages it for enhanced structural analysis:

### 3.1 Integration Point

ApexCore Mini is invoked at the ApexLite stage:

```
CandleLite Output → Feature Extraction → ApexCore Mini → Structural Scores
```

### 3.2 Outputs Used

From ApexCore Mini, QuantraVision uses:
- `regime` — To contextualize analysis
- `risk_tier` — For risk indicator display
- `chart_quality` — For confidence calibration
- `quantrascore_numeric` — For composite overlay

### 3.3 Fallback Behavior

If ApexCore Mini is not available:
- ApexLite uses a simplified rule-based fallback
- Overlays are flagged as "reduced confidence"
- Users are prompted to install ApexCore Mini

---

## 4. Safety Constraints

QuantraVision Apex operates under strict safety rules:

### 4.1 No Trading Features

The system has **zero execution capability**:
- No order placement
- No account integration
- No broker connections
- No portfolio modifications

Analysis is strictly read-only.

### 4.2 Narration-Only Cloud (Optional)

If cloud narration is enabled:
- Only text-based commentary is sent/received
- No trading signals or recommendations
- No personal data transmission
- All narration passes through content filters

### 4.3 Forbidden Phrases

The system validates all generated text against a forbidden phrase list:

```yaml
forbidden_patterns:
  - "buy now"
  - "sell now"
  - "guaranteed"
  - "profit"
  - "you should"
  - "financial advice"
  - "recommendation"
  - "sure thing"
  - "can't lose"
```

Any output containing forbidden phrases is blocked before display.

### 4.4 Disclaimer Injection

All overlays include visible disclaimers:
- "For educational purposes only"
- "Not financial advice"
- "Past performance is not indicative of future results"

---

## 5. Tier and Quota System

QuantraVision uses a tiered access model:

### 5.1 Tiers

| Tier | Analyses/Day | Cloud Narration | ApexCore Mini |
|------|--------------|-----------------|---------------|
| Free | 10 | No | No |
| Basic | 50 | Limited | Yes |
| Pro | Unlimited | Full | Yes |
| Enterprise | Unlimited | Full + Custom | Yes + Full |

### 5.2 Quota Enforcement

- Quotas are enforced locally on-device
- Quota state syncs with account server daily
- Exceeded quotas result in graceful degradation (not crashes)
- Enterprise tier supports offline quota grants

---

## 6. HUD Overlay Elements

The overlay presents information through several HUD components:

### 6.1 Score Badge

Displays the QuantraScore as a color-coded badge:
- Green (0.7–1.0): High structural quality
- Yellow (0.4–0.7): Moderate quality
- Red (0.0–0.4): Low quality / caution

### 6.2 Regime Indicator

Shows the detected market regime:
- Trending (directional arrow)
- Ranging (horizontal bars)
- Volatile (lightning bolt)
- Suppressed (pause icon)

### 6.3 Risk Level

Displays risk tier with color coding:
- Low: Green shield
- Medium: Yellow triangle
- High: Orange warning
- Extreme: Red stop sign

### 6.4 Primitive Overlays

Draws detected chart structures:
- Support/resistance lines
- Trend channels
- Pattern boundaries
- Key price levels

---

## 7. Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| End-to-end latency | <500ms | Capture to overlay |
| ApexCore Mini inference | <30ms | Model only |
| Battery impact | <5% per hour | Active use |
| Memory usage | <150MB | Peak |
| Crash rate | <0.1% | Per session |

---

## 8. Summary

QuantraVision Apex v4.2 brings real-time structural chart analysis to mobile devices through an efficient, safety-constrained pipeline. By leveraging ApexCore Mini for on-device inference and enforcing strict content filtering, the system delivers valuable insights while maintaining compliance with regulatory requirements and platform safety standards.
