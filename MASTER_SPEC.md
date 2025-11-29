# QuantraCore Apex v9.0-A — Master Specification

**Version:** 9.0-A (Alpha)  
**Last Updated:** 2025-11-29  
**Classification:** Technical Build Specification  
**Purpose:** Complete technical reference for development teams

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Core Engine](#3-core-engine)
4. [Protocol System](#4-protocol-system)
   - [4.1 Protocol Architecture](#41-protocol-architecture)
   - [4.2 Tier Protocols (T01-T80) — Complete Reference](#42-tier-protocols-t01-t80--complete-reference)
   - [4.3 Learning Protocols (LP01-LP25) — Complete Reference](#43-learning-protocols-lp01-lp25--complete-reference)
   - [4.4 MonsterRunner Protocols (MR01-MR20) — Complete Reference](#44-monsterrunner-protocols-mr01-mr20--complete-reference)
   - [4.5 Omega Directives (Ω1-Ω20) — Complete Reference](#45-omega-directives-ω1-ω20--complete-reference)
5. [ApexCore Neural Models](#5-apexcore-neural-models)
6. [ApexLab Training Pipeline](#6-apexlab-training-pipeline)
7. [Broker Layer](#7-broker-layer)
8. [EEO Engine (Entry/Exit Optimization)](#8-eeo-engine)
9. [Alpha Factory](#9-alpha-factory)
10. [MarketSimulator](#10-marketsimulator)
11. [Data Layer](#11-data-layer)
12. [Hardening Infrastructure](#12-hardening-infrastructure)
13. [Risk Engine](#13-risk-engine)
14. [Compliance Framework](#14-compliance-framework)
15. [API Reference](#15-api-reference)
16. [Google Docs Integration](#16-google-docs-integration)
17. [Configuration Reference](#17-configuration-reference)
18. [Data Schemas](#18-data-schemas)
19. [Testing Framework](#19-testing-framework)
20. [Deployment](#20-deployment)

---

## 1. System Overview

### 1.1 Purpose

QuantraCore Apex v9.0-A is an **institutional-grade, deterministic AI trading intelligence engine** designed for desktop research and backtesting. The system provides structural probabilities and technical analysis outputs while enforcing strict compliance with research-only mode.

### 1.2 Key Principles

| Principle | Description |
|-----------|-------------|
| **Determinism** | Same inputs always produce identical outputs |
| **Fail-Closed** | System blocks operations on uncertainty |
| **Local-Only Learning** | No cloud dependencies; all ML runs on-device |
| **Research-Only** | No live trading; Ω4 Compliance Override always active |
| **Rule Override** | Hardcoded rules always override ML predictions |

### 1.3 Capability Summary

- **145+ Protocols**: 80 Tier + 25 Learning + 20 MonsterRunner + 20 Omega
- **QuantraScore**: 0-100 probability-weighted composite score
- **Universal Scanner**: 7 market cap buckets × 8 scan modes
- **Offline ML**: On-device ApexCore v2 neural models
- **Paper Trading**: Alpaca integration (LIVE mode disabled)
- **Self-Learning**: Feedback loop → ApexLab → periodic retraining

### 1.4 Technology Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.11, FastAPI, Uvicorn |
| Frontend | React 18.2, Vite 5, Tailwind CSS 3.4, TypeScript |
| Machine Learning | scikit-learn (GradientBoosting), joblib |
| Numerical | NumPy, Pandas |
| Testing | pytest (1,145+ tests), vitest |
| Data | Polygon.io, Alpha Vantage, Synthetic |

---

## 2. Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        QuantraCore Apex v9.0-A                       │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   ApexDesk  │  │   FastAPI   │  │    CLI      │  │   Alpha     │ │
│  │  Frontend   │  │   Server    │  │   qapex     │  │   Factory   │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │
│         └─────────────────┴────────────────┴────────────────┘       │
│                                    │                                 │
│  ┌─────────────────────────────────┴─────────────────────────────┐  │
│  │                         ApexEngine                             │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │  │
│  │  │Microtraits│ │ Entropy  │ │  Drift   │ │Suppression│        │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘         │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │  │
│  │  │Continuation│ │  Regime  │ │QuantraScore│ │  Verdict │       │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘         │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                    │                                 │
│  ┌─────────────────────────────────┴─────────────────────────────┐  │
│  │                      Protocol System                           │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │  │
│  │  │T01-T80  │  │LP01-LP25│  │MR01-MR20│  │ Ω1-Ω20  │          │  │
│  │  │  Tier   │  │Learning │  │Monster  │  │  Omega  │          │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                    │                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │  ApexCore  │  │  ApexLab   │  │   Broker   │  │    EEO     │    │
│  │    v2      │  │    v2      │  │   Layer    │  │   Engine   │    │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘    │
│                                    │                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │   Data     │  │  Market    │  │  Hardening │  │   Google   │    │
│  │   Layer    │  │ Simulator  │  │   Infra    │  │   Docs     │    │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Directory Structure

```
quantracore-apex/
├── src/quantracore_apex/
│   ├── core/                    # ApexEngine core modules
│   ├── protocols/               # Protocol implementations
│   │   ├── tier/               # T01-T80 (80 protocols)
│   │   ├── learning/           # LP01-LP25 (25 protocols)
│   │   ├── monster_runner/     # MR01-MR20 (20 protocols)
│   │   └── omega/              # Ω1-Ω20 (20 directives)
│   ├── apexcore/               # Neural models (v2)
│   ├── apexlab/                # Training pipeline (v2)
│   ├── broker/                 # Broker adapters
│   ├── eeo_engine/             # Entry/Exit Optimization
│   ├── alpha_factory/          # 24/7 live research loop
│   ├── simulator/              # MarketSimulator chaos engine
│   ├── data_layer/             # Data providers
│   ├── hardening/              # Safety infrastructure
│   ├── risk/                   # Risk engine
│   ├── compliance/             # Regulatory compliance
│   ├── integrations/           # External integrations
│   │   └── google_docs/        # Google Docs export pipeline
│   └── server/                 # FastAPI server
├── config/                      # Configuration files
├── data/                        # Training data and caches
├── dashboard/                   # React frontend
├── tests/                       # Test suites
└── logs/                        # Audit and proof logs
```

---

## 3. Core Engine

### 3.1 ApexEngine

**Location:** `src/quantracore_apex/core/engine.py`

The ApexEngine is the primary entry point for deterministic analysis. It processes OHLCV windows and produces comprehensive `ApexResult` objects.

#### 3.1.1 Execution Flow

```
OhlcvWindow (100 bars)
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  1. compute_microtraits()  → Microtraits                 │
│  2. compute_entropy()      → EntropyMetrics              │
│  3. compute_suppression()  → SuppressionMetrics          │
│  4. compute_drift()        → DriftMetrics                │
│  5. compute_continuation() → ContinuationMetrics         │
│  6. compute_volume_metrics() → VolumeMetrics             │
│  7. classify_regime()      → Regime                      │
│  8. compute_quantrascore() → QuantraScore (0-100)        │
│  9. build_verdict()        → Verdict + RiskTier          │
│ 10. run_all_protocols()    → List[ProtocolResult]        │
│ 11. apply_omega_directives() → OmegaOverrides            │
└──────────────────────────────────────────────────────────┘
    │
    ▼
ApexResult (complete analysis)
```

#### 3.1.2 Core Modules

| Module | Description | Output |
|--------|-------------|--------|
| **Microtraits** | 15+ structural features from OHLCV | `Microtraits` dataclass |
| **Entropy** | Market chaos measurement | `EntropyMetrics` + `EntropyState` |
| **Suppression** | Signal suppression detection | `SuppressionMetrics` + `SuppressionState` |
| **Drift** | Mean-reversion pressure analysis | `DriftMetrics` + `DriftState` |
| **Continuation** | Trend continuation probability | `ContinuationMetrics` |
| **Volume** | Volume spike and climax detection | `VolumeMetrics` |
| **Regime** | Market regime classification | `Regime` enum |
| **QuantraScore** | Composite probability score | `int` (0-100) |
| **Verdict** | Structural analysis verdict | `Verdict` + `RiskTier` |

#### 3.1.3 Microtraits Features

```python
@dataclass
class Microtraits:
    trend_consistency: float    # -1 to 1, measures directional consistency
    volatility_ratio: float     # Current/historical volatility
    momentum_score: float       # 0 to 1, momentum strength
    body_ratio: float           # Body/range ratio (candle structure)
    compression_score: float    # 0 to 1, volatility compression
    volume_score: float         # 0 to 1, volume relative to average
    range_expansion: float      # Current/average range
    noise: float                # Price noise level
    volatility: float           # Absolute volatility
    wick_ratio: float           # Upper+lower wick / body
    gap_ratio: float            # Gap size / ATR
    atr_14: float               # 14-period ATR
```

#### 3.1.4 Regime Classification

| Regime | Condition | Description |
|--------|-----------|-------------|
| `trending_up` | Uptrend alignment | SMA20 > SMA50, price > both |
| `trending_down` | Downtrend alignment | SMA20 < SMA50, price < both |
| `range_bound` | Horizontal consolidation | Price oscillating around mean |
| `volatile` | High volatility | ATR > 2x historical average |
| `compressed` | Low volatility squeeze | ATR < 0.5x historical average |
| `unknown` | Insufficient data | < 50 bars of data |

---

## 4. Protocol System

### 4.1 Protocol Architecture

All protocols follow a standardized interface:

```python
def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    Run protocol analysis.
    
    Returns:
        ProtocolResult with:
        - protocol_id: str (e.g., "T01")
        - fired: bool (did protocol trigger?)
        - confidence: float (0-1)
        - signal_type: Optional[str]
        - details: Dict[str, Any]
    """
```

---

### 4.2 Tier Protocols (T01-T80) — Complete Reference

**Location:** `src/quantracore_apex/protocols/tier/`

#### 4.2.1 Trend Analysis Protocols (T01-T05)

| ID | Name | Description | Trigger Condition | Key Metrics | File |
|----|------|-------------|-------------------|-------------|------|
| **T01** | Trend Direction | Analyzes primary trend using SMA20/SMA50 alignment | Price above both SMAs with trend_consistency > 0.3 | `current_price`, `sma20`, `sma50`, `trend_consistency` | `T01.py` |
| **T02** | Trend Strength | Measures trend strength via ADX computation | ADX > 25 with combined_strength > 0.4 | `adx`, `combined_strength`, `body_consistency` | `T02.py` |
| **T03** | Trend Momentum | Rate of change analysis across multiple periods | ROC 5/10/20 aligned in same direction | `roc_5`, `roc_10`, `roc_20`, `momentum_aligned` | `T03.py` |
| **T04** | Trend Channel | Detects parallel trend channels | Price within defined channel, parallel score > 0.7 | `channel_direction`, `position_in_channel`, `channel_width` | `T04.py` |
| **T05** | MA Crossover | Moving average crossover detection | Recent SMA10/SMA20 crossover within 10 bars | `current_fast_ma`, `current_slow_ma`, `bars_since_cross` | `T05.py` |

#### 4.2.2 Volatility Structure Protocols (T06-T10)

| ID | Name | Description | Trigger Condition | Key Metrics | File |
|----|------|-------------|-------------------|-------------|------|
| **T06** | Volatility Expansion | Detects expansion from compressed states | Recent range > 1.5x prior range after compression | `expansion_ratio`, `was_compressed`, `compression_score` | `T06.py` |
| **T07** | Bollinger Squeeze | Bollinger Band width contraction | Squeeze ratio < 0.6 and near 6-month minimum | `current_bb_width`, `squeeze_ratio`, `near_minimum` | `T07.py` |
| **T08** | ATR Contraction | ATR-based volatility contraction | ATR < 0.6x historical average, declining trend | `current_atr`, `average_atr`, `contraction_ratio` | `T08.py` |
| **T09** | Range Compression | Consecutive narrowing price ranges | 5+ narrowing bars, compression > 30% | `narrowing_bars`, `compression_pct` | `T09.py` |
| **T10** | IV Structure | Structural volatility pattern analysis | Realized vol ratio < 0.6 or > 1.5 | `realized_vol`, `vol_ratio`, `vol_skew` | `T10.py` |

#### 4.2.3 Continuation Pattern Protocols (T11-T15)

| ID | Name | Description | Trigger Condition | Key Metrics | File |
|----|------|-------------|-------------------|-------------|------|
| **T11** | Bullish Continuation | Bull flag/pennant detection | Uptrend followed by consolidation | `flag_depth`, `consolidation_bars`, `trend_strength` | `T11.py` |
| **T12** | Bearish Continuation | Bear flag/pennant detection | Downtrend followed by consolidation | `flag_depth`, `consolidation_bars`, `trend_strength` | `T12.py` |
| **T13** | Flag Pattern | Classic flag consolidation | Tight parallel consolidation after impulse | `flag_width`, `impulse_strength`, `consolidation_quality` | `T13.py` |
| **T14** | Pennant Pattern | Symmetrical narrowing consolidation | Converging trendlines after impulse | `apex_proximity`, `symmetry_score`, `volume_decline` | `T14.py` |
| **T15** | Wedge Pattern | Rising/falling wedge detection | Converging trendlines with bias | `wedge_type`, `slope_convergence`, `breakout_proximity` | `T15.py` |

#### 4.2.4 Reversal Pattern Protocols (T16-T20)

| ID | Name | Description | Trigger Condition | Key Metrics | File |
|----|------|-------------|-------------------|-------------|------|
| **T16** | Double Bottom | W-bottom reversal pattern | Two lows within 3%, neckline approach | `low1`, `low2`, `neckline`, `symmetry_score` | `T16.py` |
| **T17** | Double Top | M-top reversal pattern | Two highs within 3%, neckline approach | `high1`, `high2`, `neckline`, `symmetry_score` | `T17.py` |
| **T18** | Head & Shoulders | Classic H&S reversal | Head higher than shoulders, neckline test | `left_shoulder`, `head`, `right_shoulder`, `neckline` | `T18.py` |
| **T19** | Inverse H&S | Bullish reversal pattern | Inverted H&S structure | `left_shoulder`, `head`, `right_shoulder`, `neckline` | `T19.py` |
| **T20** | Volume Climax | Exhaustion volume spike | Volume > 3x average with reversal candle | `volume_multiple`, `reversal_signal`, `price_action` | `T20.py` |

#### 4.2.5 Advanced Volatility Protocols (T21-T30)

| ID | Name | Description | Trigger Condition | Key Metrics | File |
|----|------|-------------|-------------------|-------------|------|
| **T21** | Vol Expansion v2 | Enhanced expansion detection | Multi-factor expansion confirmation | `expansion_score`, `breakout_direction`, `momentum_confirm` | `T21.py` |
| **T22** | Vol Contraction v2 | Enhanced compression detection | Multi-factor compression confirmation | `contraction_score`, `squeeze_intensity`, `coil_energy` | `T22.py` |
| **T23** | BB Squeeze v2 | Improved Bollinger analysis | Keltner inside Bollinger squeeze | `bb_width`, `keltner_inside`, `squeeze_bars` | `T23.py` |
| **T24** | ATR Breakout | ATR-based breakout confirmation | Price move > 2x ATR | `atr_multiple`, `breakout_direction`, `follow_through` | `T24.py` |
| **T25** | Keltner Channel | Keltner band analysis | Price outside Keltner bands | `price_vs_upper`, `price_vs_lower`, `channel_width` | `T25.py` |
| **T26** | HV Regime | Historical volatility regime | Clear volatility regime identification | `hv_percentile`, `regime_type`, `regime_stability` | `T26.py` |
| **T27** | Volatility Skew | Asymmetric volatility patterns | Up/down vol ratio deviation | `upside_vol`, `downside_vol`, `skew_ratio` | `T27.py` |
| **T28** | Intraday Range | Intraday range patterns | Range expansion/contraction patterns | `range_vs_average`, `time_of_day_factor` | `T28.py` |
| **T29** | Gap Volatility | Gap pattern analysis | Significant gap with vol implications | `gap_size`, `gap_fill_probability`, `vol_impact` | `T29.py` |
| **T30** | Vol Mean Reversion | Volatility normalization detection | Extended volatility likely to revert | `vol_z_score`, `reversion_probability`, `expected_vol` | `T30.py` |

#### 4.2.6 Momentum Oscillator Protocols (T31-T40)

| ID | Name | Description | Trigger Condition | Key Metrics | File |
|----|------|-------------|-------------------|-------------|------|
| **T31** | RSI Momentum | RSI overbought/oversold detection | RSI < 30 or > 70 | `rsi_value`, `rsi_slope`, `divergence_detected` | `T31.py` |
| **T32** | MACD Momentum | MACD crossover and histogram | MACD line crosses signal line | `macd_line`, `signal_line`, `histogram`, `crossover_type` | `T32.py` |
| **T33** | Stochastic | Stochastic oscillator signals | %K crosses %D in extreme zones | `percent_k`, `percent_d`, `crossover_zone` | `T33.py` |
| **T34** | Rate of Change | ROC momentum analysis | ROC exceeds threshold | `roc_value`, `roc_percentile`, `acceleration` | `T34.py` |
| **T35** | Momentum Divergence | Price-momentum divergence | Price makes new high/low, momentum doesn't | `price_direction`, `momentum_direction`, `divergence_type` | `T35.py` |
| **T36** | ADX Trend Strength | ADX-based strength measurement | ADX > 25 indicates strong trend | `adx_value`, `plus_di`, `minus_di`, `trend_quality` | `T36.py` |
| **T37** | Williams %R | Williams indicator extremes | %R in overbought/oversold zone | `williams_r`, `extreme_zone`, `reversal_signal` | `T37.py` |
| **T38** | CCI Momentum | Commodity Channel Index | CCI > 100 or < -100 | `cci_value`, `cci_trend`, `extreme_level` | `T38.py` |
| **T39** | Momentum Exhaustion | Exhaustion pattern detection | Declining momentum with price extension | `momentum_slope`, `price_extension`, `exhaustion_score` | `T39.py` |
| **T40** | MTF Alignment | Multi-timeframe momentum alignment | Momentum aligned across timeframes | `tf1_momentum`, `tf2_momentum`, `alignment_score` | `T40.py` |

#### 4.2.7 Volume Analysis Protocols (T41-T50)

| ID | Name | Description | Trigger Condition | Key Metrics | File |
|----|------|-------------|-------------------|-------------|------|
| **T41** | Volume Spike | Institutional volume detection | Volume > 2x 20-day average | `volume_multiple`, `price_impact`, `institutional_signature` | `T41.py` |
| **T42** | OBV | On-Balance Volume trend | OBV making new highs/lows | `obv_value`, `obv_trend`, `price_obv_divergence` | `T42.py` |
| **T43** | Volume Confirmation | Price-volume relationship | Volume confirms price movement | `volume_price_correlation`, `confirmation_score` | `T43.py` |
| **T44** | A/D Line | Accumulation/Distribution | A/D line divergence or confirmation | `ad_line`, `ad_trend`, `accumulation_distribution` | `T44.py` |
| **T45** | MFI | Money Flow Index extremes | MFI > 80 or < 20 | `mfi_value`, `extreme_zone`, `flow_direction` | `T45.py` |
| **T46** | VWAP Analysis | Price vs VWAP position | Significant deviation from VWAP | `price_vs_vwap`, `vwap_deviation_pct`, `vwap_trend` | `T46.py` |
| **T47** | Chaikin MF | Chaikin Money Flow | CMF > 0.25 or < -0.25 | `cmf_value`, `buying_pressure`, `selling_pressure` | `T47.py` |
| **T48** | Volume Profile | Volume distribution analysis | Price at high/low volume nodes | `volume_poc`, `value_area_high`, `value_area_low` | `T48.py` |
| **T49** | Force Index | Trend strength with volume | Force Index extreme readings | `force_index`, `trend_strength`, `volume_force` | `T49.py` |
| **T50** | Volume Climax v2 | Enhanced climax detection | Multi-factor climax confirmation | `climax_score`, `exhaustion_probability`, `reversal_setup` | `T50.py` |

#### 4.2.8 Pattern Recognition Protocols (T51-T60)

| ID | Name | Description | Trigger Condition | Key Metrics | File |
|----|------|-------------|-------------------|-------------|------|
| **T51** | Candlestick Patterns | Japanese candlestick recognition | Doji, hammer, engulfing, etc. detected | `pattern_name`, `pattern_type`, `reliability_score` | `T51.py` |
| **T52** | Double Top/Bottom v2 | Enhanced pattern detection | Pattern with volume confirmation | `pattern_type`, `symmetry`, `volume_profile` | `T52.py` |
| **T53** | H&S v2 | Enhanced H&S detection | Pattern with momentum confirmation | `pattern_quality`, `neckline_angle`, `momentum_confirm` | `T53.py` |
| **T54** | Triangle Patterns | Ascending/descending/symmetrical | Triangle consolidation identified | `triangle_type`, `apex_bars`, `breakout_probability` | `T54.py` |
| **T55** | Flag/Pennant v2 | Enhanced continuation patterns | Pattern with volume decline | `pattern_type`, `consolidation_quality`, `breakout_setup` | `T55.py` |
| **T56** | Wedge v2 | Enhanced wedge detection | Pattern with momentum divergence | `wedge_type`, `slope_convergence`, `divergence_present` | `T56.py` |
| **T57** | Cup and Handle | Classic continuation pattern | U-shaped base with handle | `cup_depth`, `handle_retrace`, `breakout_level` | `T57.py` |
| **T58** | Island Reversal | Gap-isolated reversal | Island gap pattern | `island_size`, `gap_quality`, `reversal_probability` | `T58.py` |
| **T59** | Rounding Bottom | Saucer reversal pattern | Gradual U-shaped base | `base_duration`, `symmetry_score`, `breakout_level` | `T59.py` |
| **T60** | Three Drives | Harmonic pattern | Three drives to high/low | `drive1`, `drive2`, `drive3`, `harmonic_ratio` | `T60.py` |

#### 4.2.9 Support/Resistance Protocols (T61-T70)

| ID | Name | Description | Trigger Condition | Key Metrics | File |
|----|------|-------------|-------------------|-------------|------|
| **T61** | Key Support | Support level detection | Price approaching/testing support | `support_level`, `test_count`, `strength_score` | `T61.py` |
| **T62** | Key Resistance | Resistance level detection | Price approaching/testing resistance | `resistance_level`, `test_count`, `strength_score` | `T62.py` |
| **T63** | MA Support/Resistance | Moving averages as S/R | Price interacting with key MAs | `ma_level`, `ma_period`, `bounce_score` | `T63.py` |
| **T64** | Fibonacci Levels | Fibonacci retracement analysis | Price at key Fibonacci levels | `fib_level`, `fib_ratio`, `confluence_score` | `T64.py` |
| **T65** | Pivot Points | Classic pivot analysis | Price at pivot levels | `pivot`, `r1`, `r2`, `s1`, `s2` | `T65.py` |
| **T66** | Psychological Levels | Round number analysis | Price at psychological levels | `psych_level`, `proximity_pct`, `reaction_history` | `T66.py` |
| **T67** | Breakout Confirmation | Breakout validation | Breakout with follow-through | `breakout_level`, `follow_through`, `volume_confirm` | `T67.py` |
| **T68** | False Breakout | Bull/bear trap detection | Breakout showing failure signs | `trap_type`, `failure_score`, `reversal_probability` | `T68.py` |
| **T69** | S/R Flip | Support/resistance role reversal | Previous S becomes R or vice versa | `flip_level`, `flip_confirmation`, `new_role` | `T69.py` |
| **T70** | Range Boundary | Range edge analysis | Price at range extremes | `range_high`, `range_low`, `boundary_test` | `T70.py` |

#### 4.2.10 Market Assessment Protocols (T71-T80)

| ID | Name | Description | Trigger Condition | Key Metrics | File |
|----|------|-------------|-------------------|-------------|------|
| **T71** | Trend Strength v2 | Multi-indicator strength | Multiple strength indicators aligned | `composite_strength`, `indicator_agreement` | `T71.py` |
| **T72** | Regime Classification | Market regime detection | Clear regime identification | `regime_type`, `regime_confidence`, `transition_risk` | `T72.py` |
| **T73** | Relative Performance | Historical comparison | Performance significantly different from normal | `percentile_rank`, `z_score`, `anomaly_score` | `T73.py` |
| **T74** | Price Position | Position in historical range | Price at extremes of historical range | `percentile_position`, `range_context` | `T74.py` |
| **T75** | Trend Maturity | Trend age/exhaustion analysis | Trend shows maturity or extension | `trend_duration`, `extension_score`, `exhaustion_risk` | `T75.py` |
| **T76** | Seasonality | Calendar pattern detection | Day-of-week or cyclical patterns | `seasonal_bias`, `historical_edge`, `confidence` | `T76.py` |
| **T77** | Mean Reversion | Reversion probability | Conditions favor mean reversion | `deviation_score`, `reversion_probability`, `expected_move` | `T77.py` |
| **T78** | Trend Continuation | Continuation probability | Conditions favor trend continuation | `continuation_score`, `momentum_support`, `structure_score` | `T78.py` |
| **T79** | Risk Environment | Overall risk assessment | Risk environment clearly elevated/subdued | `risk_score`, `vol_regime`, `correlation_risk` | `T79.py` |
| **T80** | Composite Signal | Multi-protocol integration | Multiple protocols aligned | `alignment_score`, `signal_count`, `conviction_level` | `T80.py` |

---

### 4.3 Learning Protocols (LP01-LP25) — Complete Reference

**Location:** `src/quantracore_apex/protocols/learning/`

These protocols generate training labels for ApexLab supervised learning.

| ID | Name | Label Type | Output Type | Classes/Range | Description | File |
|----|------|------------|-------------|---------------|-------------|------|
| **LP01** | Regime Label | `regime_class` | Classification | 6 classes | Maps regime to numeric: trending_up(0), trending_down(1), range_bound(2), volatile(3), compressed(4), unknown(5) | `LP01.py` |
| **LP02** | Volatility Band | `volatility_band` | Classification | 5 bands | Classifies volatility_ratio: very_low(0), low(1), normal(2), elevated(3), high(4) | `LP02.py` |
| **LP03** | Risk Tier | `risk_tier` | Classification | 4 tiers | Maps risk tier: low(0), medium(1), high(2), extreme(3) | `LP03.py` |
| **LP04** | Suppression State | `suppression_state` | Classification | 4 states | Maps suppression: none(0), light(1), moderate(2), heavy(3) | `LP04.py` |
| **LP05** | Entropy State | `entropy_state` | Classification | 3 states | Maps entropy: stable(0), elevated(1), chaotic(2) | `LP05.py` |
| **LP06** | Continuation Result | `continuation_result` | Classification | 3 outcomes | Classifies: likely_reverse(0), neutral(1), likely_continue(2) | `LP06.py` |
| **LP07** | Score Bucket | `score_bucket` | Classification | 5 buckets | Maps score bucket: very_low(0), low(1), neutral(2), high(3), very_high(4) | `LP07.py` |
| **LP08** | Sector Bias | `sector_bias` | Classification | 3 biases | Sector-specific bias: bearish(0), neutral(1), bullish(2) | `LP08.py` |
| **LP09** | QuantraScore Numeric | `quantrascore_numeric` | Regression | 0-100 | Raw QuantraScore as regression target | `LP09.py` |
| **LP10** | Drift State | `drift_state` | Classification | 4 states | Maps drift: none(0), mild(1), significant(2), critical(3) | `LP10.py` |
| **LP11** | Future Direction | `future_direction` | Classification | 3 directions | Predicted direction: down(0), flat(1), up(2) | `LP11.py` |
| **LP12** | Future Volatility | `future_volatility` | Classification | 3 regimes | Predicted vol regime: low(0), normal(1), high(2) | `LP12.py` |
| **LP13** | Momentum Persistence | `momentum_persistence` | Classification | 4 states | Momentum duration: fading(0), weak(1), steady(2), strong(3) | `LP13.py` |
| **LP14** | Breakout Direction | `breakout_direction` | Classification | 3 directions | Breakout direction: down(0), none(1), up(2) | `LP14.py` |
| **LP15** | Volume Confirmation | `volume_confirmation` | Classification | 5 levels | Volume confirmation: very_weak(0) to very_strong(4) | `LP15.py` |
| **LP16** | Pattern Quality | `pattern_quality` | Classification | 3 tiers | Pattern quality: poor(0), fair(1), excellent(2) | `LP16.py` |
| **LP17** | S/R Proximity | `sr_proximity` | Classification | 4 positions | Position vs S/R: near_support(0), mid(1), near_resistance(2), breakout(3) | `LP17.py` |
| **LP18** | Trend Exhaustion | `trend_exhaustion` | Classification | 3 states | Exhaustion level: fresh(0), mature(1), exhausted(2) | `LP18.py` |
| **LP19** | Mean Reversion | `mean_reversion` | Classification | 3 potentials | Reversion potential: low(0), medium(1), high(2) | `LP19.py` |
| **LP20** | Breakout Authenticity | `breakout_authenticity` | Classification | 3 levels | Breakout quality: false(0), uncertain(1), authentic(2) | `LP20.py` |
| **LP21** | Institutional Activity | `institutional_activity` | Classification | 3 levels | Institutional presence: none(0), some(1), heavy(2) | `LP21.py` |
| **LP22** | Market Phase | `market_phase` | Classification | 4 phases | Market phase: accumulation(0), markup(1), distribution(2), markdown(3) | `LP22.py` |
| **LP23** | Risk-Adjusted Score | `risk_adjusted_score` | Classification | 3 tiers | Risk-adjusted quality: poor(0), fair(1), excellent(2) | `LP23.py` |
| **LP24** | Time-to-Event | `time_to_event` | Classification | 3 horizons | Event proximity: imminent(0), near(1), distant(2) | `LP24.py` |
| **LP25** | Composite Conviction | `composite_conviction` | Classification | 3 levels | Overall conviction: low(0), medium(1), high(2) | `LP25.py` |

---

### 4.4 MonsterRunner Protocols (MR01-MR20) — Complete Reference

**Location:** `src/quantracore_apex/protocols/monster_runner/`

These protocols detect explosive move precursors for potential multi-bagger situations.

| ID | Name | Description | Detection Logic | Risk Multiplier | Key Metrics | File |
|----|------|-------------|-----------------|-----------------|-------------|------|
| **MR01** | Compression Explosion | ATR contraction + BB squeeze before breakout | ATR < 0.5x avg, BB squeeze > 80% | 2.0x | `compression_score`, `atr_contraction_ratio`, `bollinger_squeeze`, `explosion_probability` | `MR01.py` |
| **MR02** | Volume Anomaly | Unusual volume patterns indicating accumulation | Volume distribution anomaly detection | 1.5x | `anomaly_score`, `volume_distribution`, `accumulation_signature` | `MR02.py` |
| **MR03** | Vol Regime Shift | Volatility regime transition detection | Regime change from low to high vol | 1.8x | `regime_before`, `regime_after`, `transition_score` | `MR03.py` |
| **MR04** | Institutional Footprint | Large block trade detection | Volume + price action signature | 1.5x | `block_score`, `institutional_probability`, `size_estimate` | `MR04.py` |
| **MR05** | MTF Alignment | Multi-timeframe alignment for explosive moves | All timeframes aligned bullish/bearish | 1.3x | `tf_alignment`, `alignment_strength`, `momentum_confirm` | `MR05.py` |
| **MR06** | Bollinger Breakout | Post-squeeze Bollinger Band breakout | BB squeeze followed by band breakout | 2.5x | `breakout_score`, `squeeze_depth`, `breakout_direction`, `band_width` | `MR06.py` |
| **MR07** | Volume Explosion | Simultaneous volume + price explosion | Volume > 3x and price > 2x ATR | 3.0x | `explosion_score`, `volume_multiple`, `price_change_pct`, `institutional_signature` | `MR07.py` |
| **MR08** | Earnings Gap | Gap-and-go after earnings | Large gap with follow-through | 2.5x | `gap_size`, `follow_through`, `volume_confirm` | `MR08.py` |
| **MR09** | VWAP Breakout | VWAP breakout with volume | Price breaks VWAP with 2x volume | 1.8x | `vwap_score`, `vwap_deviation`, `volume_confirmation` | `MR09.py` |
| **MR10** | NR7 Breakout | Narrow Range 7 breakout | Smallest 7-day range followed by expansion | 2.0x | `nr7_score`, `is_nr7`, `range_rank`, `breakout_detected` | `MR10.py` |
| **MR11** | Short Squeeze | Short/gamma squeeze detection | High short interest + price acceleration | 4.0x | `squeeze_score`, `price_acceleration`, `volume_surge`, `parabolic_move` | `MR11.py` |
| **MR12** | Crypto Pump | Parabolic crypto move detection | > 20% intraday move with 5x volume | 3.5x | `pump_score`, `intraday_move`, `volume_ratio`, `is_pump` | `MR12.py` |
| **MR13** | News Catalyst | Gap move from news event | Large gap with volume spike | 2.5x | `catalyst_score`, `gap_magnitude`, `gap_direction`, `volume_spike` | `MR13.py` |
| **MR14** | Fractal Explosion | 20-day high/low breakout with momentum | New 20-day high/low with momentum confirm | 1.5x | `fractal_score`, `is_new_high`, `is_new_low`, `breakout_strength` | `MR14.py` |
| **MR15** | 100% Day | Intraday doubler detection | Price doubles intraday | 5.0x | `extreme_score`, `total_return`, `intraday_return`, `is_doubler` | `MR15.py` |
| **MR16** | Parabolic Phase 3 | Blow-off top detection | Late-stage parabolic acceleration | 4.0x | `parabolic_score`, `five_day_return`, `acceleration_rate` | `MR16.py` |
| **MR17** | Meme Frenzy | Meme stock momentum detection | High volatility + retail volume patterns | 3.5x | `frenzy_score`, `volatility_ratio`, `volume_intensity` | `MR17.py` |
| **MR18** | Gamma Ramp | Options gamma squeeze | Volatility spike + intraday range expansion | 3.0x | `gamma_score`, `volatility_spike`, `intraday_range_ratio` | `MR18.py` |
| **MR19** | FOMO Cascade | FOMO momentum cascade | Cumulative return > 50% in 10 days | 2.5x | `fomo_score`, `cumulative_return_10d`, `volume_growth` | `MR19.py` |
| **MR20** | Nuclear Runner | Extreme multi-day runner (3x+ in 10 days) | > 200% return in 10 trading days | 5.0x | `nuclear_score`, `multiplier`, `total_return_10d` | `MR20.py` |

---

### 4.5 Omega Directives (Ω1-Ω20) — Complete Reference

**Location:** `src/quantracore_apex/protocols/omega/omega.py`

Safety override protocols that enforce fail-closed behavior.

#### Omega Levels

| Level | Behavior | Description |
|-------|----------|-------------|
| `INACTIVE` | Directive not triggered | Normal operation |
| `ADVISORY` | Warning issued, signal allowed | Alert only, no blocking |
| `ENFORCED` | Signal modified or restricted | Active restriction |
| `LOCKED` | All signals blocked | Full halt |

#### Complete Omega Directive Reference

| ID | Name | Trigger Condition | Level | Description | Metadata |
|----|------|-------------------|-------|-------------|----------|
| **Ω1** | Hard Safety Lock | `risk_tier == EXTREME` OR `quantrascore < 5` OR `quantrascore > 98` | LOCKED | Blocks all signals when extreme conditions detected | `risk_tier`, `quantrascore` |
| **Ω2** | Entropy Override | `entropy_state == CHAOTIC` OR `combined_entropy > 0.9` | ENFORCED | Forces caution on high market entropy/chaos | `entropy_state`, `combined_entropy` |
| **Ω3** | Drift Override | `drift_state == CRITICAL` OR (`drift_state == SIGNIFICANT` AND `reversion_pressure < 0.3`) | ENFORCED | Forces caution on critical price drift | `drift_state`, `drift_magnitude`, `reversion_pressure` |
| **Ω4** | Compliance Override | Always active | ENFORCED | Ensures all outputs framed as structural analysis, not trading recommendations | `compliance_note`, `action_type` |
| **Ω5** | Suppression Lock | `suppression_state == HEAVY` | ENFORCED | Blocks signals under heavy suppression conditions | `suppression_state`, `suppression_level` |
| **Ω6** | Volatility Cap | `volatility > 0.04` (4%) | ENFORCED | Limits exposure in high volatility environments | `volatility`, `volatility_ratio` |
| **Ω7** | Momentum Divergence | Price/momentum divergence > 40% | ENFORCED | Detects price/volume divergence warning | `price_direction`, `momentum_direction`, `divergence_pct` |
| **Ω8** | BB Squeeze Warning | `compression_score > 0.8` (80%) | ADVISORY | Signals potential explosive breakout imminent | `compression_score`, `squeeze_duration` |
| **Ω9** | MACD Reversal | MACD histogram reversal detected | ADVISORY | Identifies momentum histogram reversals | `histogram_change`, `reversal_type` |
| **Ω10** | Fear Spike | `atr_ratio > 1.5` AND `fear_score > 0.06` | LOCKED | Triggers on elevated ATR/fear levels | `atr_ratio`, `fear_score` |
| **Ω11** | RSI Extreme | `RSI < 20` OR `RSI > 80` | ENFORCED | Blocks signals at extreme RSI levels | `rsi_value`, `extreme_type` |
| **Ω12** | Volume Spike Alert | `volume > 3x average` | ADVISORY | Detects abnormal volume surges | `volume_multiple`, `average_volume` |
| **Ω13** | Trend Weakness | `noise > threshold` AND `trend_consistency < 0.3` | ADVISORY | Signals weak/uncertain trend conditions | `noise`, `trend_consistency` |
| **Ω14** | Gap Risk | `gap_ratio > 0.03` (3%) | ENFORCED | Blocks signals after large overnight gaps | `gap_ratio`, `gap_direction` |
| **Ω15** | Tail Risk Lock | Intraday drop > 5% | LOCKED | Engages on severe intraday price drops | `intraday_return`, `drop_magnitude` |
| **Ω16** | Overnight Drift | Open-to-close move > 3% | ADVISORY | Detects large open-to-close price moves | `oc_return`, `drift_direction` |
| **Ω17** | Fractal Chaos | Fractal dimension indicates chaos | ENFORCED | Signals unstable price structure | `fractal_dimension`, `chaos_score` |
| **Ω18** | Liquidity Void | `volume_score < 0.2` (20%) | ENFORCED | Blocks signals on low liquidity conditions | `volume_score`, `average_volume` |
| **Ω19** | Correlation Breakdown | Cross-asset correlation anomaly | ENFORCED | Detects unusual cross-asset correlations | `correlation_z_score`, `affected_assets` |
| **Ω20** | Nuclear Killswitch | Data integrity failure OR system anomaly | LOCKED | Final safety gate for catastrophic conditions | `integrity_check`, `anomaly_type` |

---

## 5. ApexCore Neural Models

### 5.1 Architecture Overview

**Location:** `src/quantracore_apex/apexcore/`

ApexCore v2 is a multi-head neural assistant using scikit-learn Gradient Boosting models.

### 5.2 Model Variants

| Variant | Estimators | Max Depth | Learning Rate | Use Case |
|---------|------------|-----------|---------------|----------|
| `BIG` | 100 | 6 | 0.1 | Full precision, research |
| `MINI` | 30 | 3 | 0.15 | Lightweight, mobile |

### 5.3 Prediction Heads

| Head | Type | Output | Description |
|------|------|--------|-------------|
| `quantra_score` | Regression | 0-100 | Predicted QuantraScore |
| `runner_prob` | Binary Classification | 0-1 | Probability of runner status |
| `quality_tier` | Multi-class | A+, A, B, C, D | Trade quality classification |
| `avoid_trade` | Binary Classification | 0-1 | Probability should avoid trade |
| `regime` | Multi-class | 5 classes | Market regime prediction |

### 5.4 Feature Encoding

**115-dimensional protocol vector:**
- 80 Tier protocol results (fired: 0/1, confidence: 0-1)
- 15 Microtraits features
- 10 Derived features (ratios, composites)
- 10 Context features (sector, market cap, timeframe)

### 5.5 Ensemble Architecture

```python
class ApexCoreV2Ensemble:
    """
    Ensemble of N models with uncertainty quantification.
    
    Provides:
    - Mean predictions across ensemble
    - Disagreement metrics for fail-closed behavior
    - Bootstrap training for diversity
    """
    ensemble_size: int = 3
    members: List[ApexCoreV2Model]
    
    def predict_with_uncertainty(self, features: np.ndarray) -> Dict:
        predictions = [m.predict(features) for m in self.members]
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        disagreement = std_pred / (mean_pred + 1e-8)
        return {"prediction": mean_pred, "uncertainty": std_pred, "disagreement": disagreement}
```

### 5.6 Model Persistence

```
data/training/models/
├── apexcore_demo.pkl          # Trained model (joblib)
├── apexcore_demo_metadata.json # Training metadata
└── ensemble/
    ├── member_0.joblib
    ├── member_1.joblib
    ├── member_2.joblib
    └── manifests/
        └── latest.json        # Manifest with SHA256 hashes
```

---

## 6. ApexLab Training Pipeline

### 6.1 Overview

**Location:** `src/quantracore_apex/apexlab/`

ApexLab v2 provides institutional-grade label generation and dataset building.

### 6.2 Training Row Schema

```python
@dataclass
class ApexLabV2Row:
    # Meta Fields
    symbol: str
    event_time: datetime
    timeframe: str
    engine_snapshot_id: str
    scanner_snapshot_id: str
    
    # Structural Inputs (from ApexEngine)
    quantra_score: float
    risk_tier: str
    entropy_band: str
    suppression_state: str
    regime_type: str
    volatility_band: str
    liquidity_band: str
    protocol_ids: List[str]
    protocol_vector: List[float]  # 115-dimensional
    
    # Future Outcomes (labels)
    ret_1d: float
    ret_3d: float
    ret_5d: float
    ret_10d: float
    max_runup_5d: float
    max_drawdown_5d: float
    time_to_peak_5d: int
    
    # Quality Labels
    future_quality_tier: str  # A_PLUS, A, B, C, D
    hit_runner_threshold: int  # 0 or 1
    hit_monster_runner_threshold: int  # 0 or 1
    
    # Safety Labels
    avoid_trade: int  # 0 or 1
    
    # Context Labels
    regime_label: str
    sector_regime: str
```

### 6.3 Training Configuration

```python
@dataclass
class TrainingConfig:
    variant: str = "big"           # "big" or "mini"
    ensemble_size: int = 3         # Number of ensemble members
    random_state: int = 42         # For reproducibility
    val_ratio: float = 0.2         # Validation split ratio
    bootstrap: bool = True         # Bootstrap sampling for ensemble
    
    # Loss weights for multi-task learning
    loss_weight_quantra_score: float = 1.0
    loss_weight_runner: float = 1.0
    loss_weight_quality_tier: float = 0.5
    loss_weight_avoid_trade: float = 1.0
    loss_weight_regime: float = 0.5
    
    # Class balancing
    balance_regimes: bool = True
    balance_runners: bool = True
    hard_negative_weight: float = 2.0  # Weight for false positives
```

### 6.4 Walk-Forward Validation

Time-aware splits prevent look-ahead bias:

```
Fold 1: [Train 0-60%] [Val 60-70%]
Fold 2: [Train 0-70%] [Val 70-80%]
Fold 3: [Train 0-80%] [Val 80-90%]
Fold 4: [Train 0-90%] [Val 90-100%]
```

### 6.5 Training Data Sources

| Source | Description | Location | Count |
|--------|-------------|----------|-------|
| Feedback Loop | Live trade outcomes | `data/apexlab/feedback_samples.json` | 107+ |
| Chaos Simulation | MarketSimulator outputs | `data/apexlab/chaos_simulation_samples.json` | 135+ |
| Historical Backtest | Offline backtesting | `data/apexlab/historical/` | Variable |
| **Total** | Combined samples | - | 242+ |

---

## 7. Broker Layer

### 7.1 Architecture

**Location:** `src/quantracore_apex/broker/`

```
┌─────────────────────────────────────────────────────────┐
│                    BrokerRouter                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │  RESEARCH   │  │   PAPER     │  │    LIVE     │      │
│  │  NullAdapter│  │ AlpacaPaper │  │  DISABLED   │      │
│  │             │  │ PaperSim    │  │             │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────────────────────┘
```

### 7.2 Execution Modes

| Mode | Adapter | Behavior |
|------|---------|----------|
| `RESEARCH` | NullAdapter | Orders logged only, no execution |
| `PAPER` | AlpacaPaperAdapter or PaperSimAdapter | Simulated execution |
| `LIVE` | **DISABLED** | Raises RuntimeError |

### 7.3 Broker Adapters

#### 7.3.1 NullAdapter
**File:** `src/quantracore_apex/broker/adapters/null_adapter.py`

```python
class NullAdapter(BrokerAdapter):
    """
    No-op adapter for research mode.
    All orders are logged but not executed.
    """
    def place_order(self, order: OrderTicket) -> ExecutionResult:
        logger.info(f"[RESEARCH] Order logged: {order}")
        return ExecutionResult(status=OrderStatus.NEW, ...)
```

#### 7.3.2 PaperSimAdapter
**File:** `src/quantracore_apex/broker/adapters/papersim_adapter.py`

```python
class PaperSimAdapter(BrokerAdapter):
    """
    Internal paper trading simulator.
    
    - MARKET orders fill immediately at last price
    - LIMIT orders fill if price crosses limit
    - Tracks positions and P&L in memory
    - NO external HTTP calls
    """
    initial_cash: float = 100_000.0
    positions: Dict[str, Position]
    orders: List[Order]
```

#### 7.3.3 AlpacaPaperAdapter
**File:** `src/quantracore_apex/broker/adapters/alpaca_adapter.py`

```python
class AlpacaPaperAdapter(BrokerAdapter):
    """
    Alpaca Paper Trading Adapter.
    
    Connects to paper-api.alpaca.markets
    Requires: ALPACA_PAPER_API_KEY, ALPACA_PAPER_API_SECRET
    """
    base_url: str = "https://paper-api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"
```

### 7.4 Order Types

| Type | Description | Execution |
|------|-------------|-----------|
| `MARKET` | Immediate execution | At current market price |
| `LIMIT` | Price-contingent | At limit price or better |
| `STOP` | Trigger market order | When stop price reached |
| `STOP_LIMIT` | Trigger limit order | When stop price reached |

### 7.5 Broker Risk Checks (9 Checks)

**File:** `src/quantracore_apex/broker/risk_engine.py`

| # | Check | Description | Default Limit |
|---|-------|-------------|---------------|
| 1 | Max Notional | Per-symbol notional limit | $100,000 |
| 2 | Max Positions | Total position count limit | 10 |
| 3 | Short Selling | Block/allow short sales | Allowed |
| 4 | Order Size | Maximum shares per order | 10,000 |
| 5 | Daily Drawdown | Daily loss limit | 5% |
| 6 | Sector Concentration | Sector exposure limit | 40% |
| 7 | Kill Switch | Global halt check | Active |
| 8 | Mode Validation | Verify paper/research mode | Required |
| 9 | Compliance | Regulatory compliance check | Active |

---

## 8. EEO Engine

### 8.1 Overview

**Location:** `src/quantracore_apex/eeo_engine/`

Entry/Exit Optimization Engine calculates optimal trading plans.

### 8.2 Entry Strategies

| Strategy | Condition | Behavior |
|----------|-----------|----------|
| `BASELINE` | Default | Standard limit orders at mid-price |
| `HIGH_VOLATILITY` | Vol band = HIGH | Conservative limits, scaled entries |
| `LOW_LIQUIDITY` | Liquidity = LOW | Inside-spread limits, smaller size |
| `RUNNER_ANTICIPATION` | High runner prob | Slightly aggressive entries |
| `ZDE_AWARE` | ZDE research label | Tightened zones, fast execution |

### 8.3 Exit Components

| Component | Description | Configuration |
|-----------|-------------|---------------|
| **Protective Stop** | ATR-based, volatility-adjusted | 1.5-3x ATR from entry |
| **Profit Target T1** | First target at 50% median move | 50% position |
| **Profit Target T2** | Second target at 80% max move | Remaining position |
| **Trailing Stop** | ATR-based or percentage-based | 2x ATR or 3% |
| **Time Exit** | Maximum holding period | 5-20 bars |
| **Abort Conditions** | Early exit triggers | Omega locks, regime change |

### 8.4 Position Sizing

```python
def calculate_position_size(
    account_equity: float,
    risk_fraction: float,  # Default 0.01 (1%)
    entry_price: float,
    stop_price: float,
    liquidity_score: float
) -> float:
    """
    Fixed-fraction risk sizing:
    size = (account_equity * risk_fraction) / stop_distance
    
    Limits applied:
    - Max 20% of available capital
    - Max notional per symbol
    - Reduced by 30% for low liquidity
    """
    stop_distance = abs(entry_price - stop_price)
    risk_amount = account_equity * risk_fraction
    base_size = risk_amount / stop_distance
    
    # Apply limits
    max_capital_pct = account_equity * 0.20
    if base_size * entry_price > max_capital_pct:
        base_size = max_capital_pct / entry_price
    
    # Liquidity adjustment
    if liquidity_score < 0.3:
        base_size *= 0.7
    
    return base_size
```

### 8.5 Entry/Exit Plan Output

```python
@dataclass
class EntryExitPlan:
    symbol: str
    direction: SignalDirection  # LONG or SHORT
    entry_mode: EntryMode
    exit_mode: ExitMode
    size_notional: float
    base_entry: BaseEntry
    scaled_entries: List[ScaledEntry]
    protective_stop: ProtectiveStop
    profit_targets: List[ProfitTarget]
    trailing_stop: TrailingStop
    time_based_exit: TimeBasedExit
    abort_conditions: List[AbortCondition]
    source_signal_id: str
    created_at: datetime
```

---

## 9. Alpha Factory

### 9.1 Overview

**Location:** `src/quantracore_apex/alpha_factory/`

24/7 live research loop with self-learning capabilities.

### 9.2 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Alpha Factory Loop                     │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐           │
│  │  Polygon  │  │  Binance  │  │ Synthetic │           │
│  │    WS     │  │    WS     │  │   Test    │           │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘           │
│        └───────────────┼───────────────┘                 │
│                        ▼                                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Price Buffer (100 bars)             │    │
│  └─────────────────────┬───────────────────────────┘    │
│                        ▼                                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │                  ApexEngine                      │    │
│  │  → Microtraits → Protocols → QuantraScore       │    │
│  └─────────────────────┬───────────────────────────┘    │
│                        ▼                                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Portfolio Engine                    │    │
│  │  → Position Management → Rebalancing            │    │
│  └─────────────────────┬───────────────────────────┘    │
│                        ▼                                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │             Feedback Loop                        │    │
│  │  → Entry Context → Exit Outcome → Training      │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### 9.3 Self-Learning Pipeline

```
Trade Entry
    │
    ├── Capture Context
    │   ├── QuantraScore (0-100)
    │   ├── Protocol flags (T01-T80)
    │   ├── Omega flags (Ω1-Ω20)
    │   ├── Microtraits snapshot
    │   └── ApexCore prediction
    │
Trade Exit
    │
    ├── Record Outcome
    │   ├── P&L (dollars + percent)
    │   ├── Holding time (minutes)
    │   ├── Max favorable excursion
    │   ├── Max adverse excursion
    │   └── Exit reason
    │
    ▼
Completed Trade
    │
    ├── Compute Label
    │   ├── STRONG_WIN (≥5%)
    │   ├── WIN (≥2%)
    │   ├── MARGINAL_WIN (≥0.5%)
    │   ├── SCRATCH (±0.5%)
    │   ├── LOSS (≥-2%)
    │   └── STRONG_LOSS (<-2%)
    │
    ▼
ApexLab Training Sample
    │
    └── Periodic Retraining (batch threshold: 50 samples)
```

### 9.4 Configuration

```python
class AlphaFactoryLoop:
    initial_cash: float = 1_000_000.0
    equity_symbols: List[str] = ["AAPL", "NVDA", "TSLA", "SPY"]
    crypto_symbols: List[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    min_score: int = 60  # Minimum QuantraScore to log
    rebalance_interval: int = 3600  # seconds
    retrain_batch_threshold: int = 50  # samples before retrain
```

---

## 10. MarketSimulator

### 10.1 Overview

**Location:** `src/quantracore_apex/simulator/`

Chaos training engine for stress-testing protocols.

### 10.2 Chaos Scenarios — Complete Reference

| Scenario | Description | Typical Drop/Gain | Volume Pattern | Risk Mult | File |
|----------|-------------|-------------------|----------------|-----------|------|
| `FLASH_CRASH` | Rapid 10-30% drop in minutes followed by potential recovery | -15% avg | 3-10x explosion | 3.0x | `scenarios.py` |
| `VOLATILITY_SPIKE` | 3-5x volatility expansion event | ±10% | 2-3x increase | 2.5x | `scenarios.py` |
| `GAP_EVENT` | Large overnight gap (earnings, news) | ±8% | Morning spike | 2.0x | `scenarios.py` |
| `LIQUIDITY_VOID` | Extremely thin order book, wide spreads | Variable | Erratic | 4.0x | `scenarios.py` |
| `MOMENTUM_EXHAUSTION` | Blow-off top or bottom pattern | Variable | Climax then fade | 2.0x | `scenarios.py` |
| `SQUEEZE` | Short squeeze or gamma squeeze | +30-100% | Massive | 4.5x | `scenarios.py` |
| `CORRELATION_BREAKDOWN` | Cross-asset correlation breakdown | Variable | Mixed | 3.0x | `scenarios.py` |
| `BLACK_SWAN` | Extreme tail event, unprecedented | -30-50% | Panic | 5.0x | `scenarios.py` |
| `NORMAL` | Baseline market conditions | ±1% | Average | 1.0x | `scenarios.py` |
| `CHOPPY` | Range-bound volatility, whipsaws | ±3% | Normal | 1.5x | `scenarios.py` |

### 10.3 Simulation Configuration

```python
@dataclass
class SimulationConfig:
    symbol: str = "SIM"
    initial_price: float = 100.0
    num_bars: int = 100
    bar_interval_minutes: int = 1
    scenario_type: ScenarioType = ScenarioType.FLASH_CRASH
    intensity: float = 1.0  # Multiplier for scenario severity
    random_seed: Optional[int] = None  # For reproducibility
    extra_params: Dict[str, Any] = field(default_factory=dict)
```

### 10.4 Usage Example

```python
from src.quantracore_apex.simulator import MarketSimulator, SimulationConfig, ScenarioType

config = SimulationConfig(
    scenario_type=ScenarioType.FLASH_CRASH,
    num_bars=100,
    intensity=1.5,
    random_seed=42
)

simulator = MarketSimulator()
result = simulator.run(config)

print(f"Max Drawdown: {result.max_drawdown_pct:.2f}%")
print(f"Max Runup: {result.max_runup_pct:.2f}%")
print(f"Bars generated: {len(result.bars)}")
```

---

## 11. Data Layer

### 11.1 Architecture

**Location:** `src/quantracore_apex/data_layer/`

### 11.2 Data Adapters

| Adapter | Source | API Key | Priority | Description |
|---------|--------|---------|----------|-------------|
| `PolygonAdapter` | Polygon.io | `POLYGON_API_KEY` | 1 | Primary production data |
| `AlphaVantageAdapter` | Alpha Vantage | `ALPHA_VANTAGE_API_KEY` | 2 | Backup provider |
| `SyntheticAdapter` | Generated | None | 3 | Testing/demo fallback |

### 11.3 Data Client

```python
from src.quantracore_apex.data_layer import DataClient

client = DataClient()

# Automatic fallback: Polygon → Alpha Vantage → Synthetic
bars = client.fetch_ohlcv(
    symbol="AAPL",
    start=datetime(2024, 1, 1),
    end=datetime(2024, 12, 31),
    timeframe="1d"
)
```

### 11.4 WebSocket Feeds

| WebSocket | Provider | Assets | Use Case |
|-----------|----------|--------|----------|
| `polygon_ws.py` | Polygon.io | US Equities | Alpha Factory equities |
| `binance_ws.py` | Binance | Crypto pairs | Alpha Factory crypto |

### 11.5 Caching

```
data/
├── cache/              # General cache
├── polygon_cache/      # Polygon-specific cache
└── historical/         # Historical data archive
```

---

## 12. Hardening Infrastructure

### 12.1 Overview

**Location:** `src/quantracore_apex/hardening/`

### 12.2 Components

| Component | File | Description |
|-----------|------|-------------|
| Config Validator | `config_validator.py` | Validates all configuration files |
| Incident Logger | `incident_logger.py` | Logs safety incidents to JSONL |
| Kill Switch | `kill_switch.py` | Global halt mechanism |
| Manifest | `manifest.py` | Protocol integrity verification |
| Mode Enforcer | `mode_enforcer.py` | Enforces research-only mode |

### 12.3 Kill Switch

**Triggers:**

| Reason | Threshold | Level |
|--------|-----------|-------|
| `DAILY_DRAWDOWN_EXCEEDED` | -5% daily | Auto |
| `BROKER_ERROR_RATE_HIGH` | 20% error rate | Auto |
| `RISK_VIOLATIONS_SPIKE` | 10 violations | Auto |
| `DATA_FEED_FAILURE` | Connection lost | Auto |
| `MODEL_FAILURE` | Prediction error | Auto |
| `DETERMINISM_FAILURE` | Hash mismatch | Auto |
| `MANUAL` | Operator triggered | Manual |
| `EXTERNAL_SIGNAL` | External API signal | External |

**State Persistence:**

```json
// config/kill_switch_state.json
{
  "engaged": false,
  "reason": null,
  "engaged_at": null,
  "engaged_by": "",
  "auto_flatten_positions": false,
  "reset_at": null,
  "reset_by": ""
}
```

### 12.4 Protocol Manifest

```yaml
# config/protocol_manifest.yaml
version: "9.0-A"
protocols:
  tier:
    count: 80
    hash: "sha256:abc123..."
  learning:
    count: 25
    hash: "sha256:def456..."
  monster_runner:
    count: 20
    hash: "sha256:ghi789..."
  omega:
    count: 20
    hash: "sha256:jkl012..."
```

---

## 13. Risk Engine

### 13.1 Overview

**Location:** `src/quantracore_apex/risk/engine.py`

### 13.2 Risk Assessment Structure

```python
@dataclass
class RiskAssessment:
    symbol: str
    timestamp: datetime
    
    # Individual risk scores (0-1)
    volatility_risk: float
    spread_risk: float
    regime_risk: float
    entropy_risk: float
    drift_risk: float
    suppression_risk: float
    fundamental_risk: float
    
    # Aggregate
    composite_risk: float
    risk_tier: str  # low, medium, high, extreme
    
    # Permission
    permission: RiskPermission  # ALLOW, DENY, RESTRICTED
    override_code: Optional[str]
    denial_reasons: List[str]
```

### 13.3 Risk Weights

| Factor | Weight | Description |
|--------|--------|-------------|
| Volatility | 0.25 | Current vs historical volatility |
| Regime | 0.20 | Market regime risk |
| Entropy | 0.15 | Market chaos level |
| Drift | 0.10 | Price drift from mean |
| Suppression | 0.10 | Signal suppression |
| Spread | 0.10 | Bid-ask spread |
| Fundamental | 0.10 | Earnings/event proximity |

### 13.4 Risk Thresholds

| Tier | Composite Score | Permission |
|------|-----------------|------------|
| `low` | 0.0 - 0.3 | ALLOW |
| `medium` | 0.3 - 0.5 | ALLOW (caution) |
| `high` | 0.5 - 0.7 | RESTRICTED |
| `extreme` | 0.7 - 1.0 | DENY |

---

## 14. Compliance Framework

### 14.1 Overview

**Location:** `src/quantracore_apex/compliance/`

### 14.2 Test Categories

| Category | File | Count | Focus |
|----------|------|-------|-------|
| Determinism | `test_determinism_verification.py` | 25+ | Same input → same output |
| Backtesting | `test_backtesting_validation.py` | 30+ | No look-ahead bias |
| Market Abuse | `test_market_abuse_detection.py` | 35+ | Manipulation prevention |
| Risk Controls | `test_risk_controls.py` | 40+ | Risk limit enforcement |
| Stress Testing | `test_stress_testing.py` | 33+ | Extreme condition handling |
| **Total Regulatory** | - | **163+** | - |

### 14.3 Compliance Note

Every output includes:

```python
compliance_note = (
    "This analysis represents structural probabilities for educational "
    "and research purposes only. It does not constitute investment advice, "
    "trading recommendations, or solicitation to trade."
)
```

---

## 15. API Reference

### 15.1 FastAPI Server

**Location:** `src/quantracore_apex/server/app.py`  
**Port:** 8000

### 15.2 Core Endpoints

| Endpoint | Method | Description | Request | Response |
|----------|--------|-------------|---------|----------|
| `/health` | GET | Health check | - | `{"status": "ok"}` |
| `/analyze` | POST | Run ApexEngine analysis | `OhlcvWindow` | `ApexResult` |
| `/scan/{symbol}` | GET | Quick symbol scan | `symbol`, `timeframe` | `ScanResult` |
| `/protocols/tier` | GET | List Tier protocols | - | `List[ProtocolInfo]` |
| `/protocols/omega` | GET | List Omega directives | - | `List[OmegaInfo]` |

### 15.3 Broker Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/broker/status` | GET | Broker connection status |
| `/broker/account` | GET | Account information |
| `/broker/positions` | GET | Current positions |
| `/broker/orders` | GET | Open orders |
| `/broker/orders` | POST | Place order |

### 15.4 Google Docs Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/google-docs/status` | GET | Connection status |
| `/google-docs/export/investor-report` | POST | Generate investor report |
| `/google-docs/export/due-diligence` | POST | Generate DD package |
| `/google-docs/export/trade-log` | POST | Export trade history |
| `/google-docs/documents` | GET | List exported documents |
| `/google-docs/journal/entry` | POST | Add journal entry |
| `/google-docs/journal/today` | GET | Get today's journal |
| `/google-docs/journal/list` | GET | List all journals |
| `/google-docs/investor-update/monthly` | POST | Monthly update |

---

## 16. Google Docs Integration

### 16.1 Overview

**Location:** `src/quantracore_apex/integrations/google_docs/`

Automated export pipeline for investor and acquirer documentation.

### 16.2 Components

| Module | Purpose |
|--------|---------|
| `client.py` | Google Docs API wrapper |
| `automated_pipeline.py` | Metrics collection and export |
| `investor_updates.py` | Investor update generation |
| `trade_journal.py` | Trade journal management |

### 16.3 Export Types

| Type | Content | Frequency |
|------|---------|-----------|
| **Investor Report** | Performance metrics, equity curve, risk stats | Daily/Weekly/Monthly |
| **Due Diligence** | System architecture, test results, compliance | On-demand |
| **Trade Log** | Complete trade history with outcomes | On-demand |
| **Trade Journal** | Daily entries with research notes | Daily |
| **Monthly Update** | Monthly performance summary | Monthly |

### 16.4 Performance Metrics

```python
@dataclass
class PerformanceMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    best_trade: float
    worst_trade: float
    average_hold_time_minutes: float
    sharpe_ratio: float
    max_drawdown: float
```

---

## 17. Configuration Reference

### 17.1 Core Configuration Files

| File | Purpose |
|------|---------|
| `config/mode.yaml` | Execution mode settings |
| `config/broker.yaml` | Broker layer configuration |
| `config/scan_modes.yaml` | Scanner configuration |
| `config/symbol_universe.yaml` | Symbol universe definitions |
| `config/protocol_manifest.yaml` | Protocol integrity hashes |
| `config/kill_switch_state.json` | Kill switch state |

### 17.2 Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `POLYGON_API_KEY` | Polygon.io data access | For live data |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage backup | Optional |
| `ALPACA_PAPER_API_KEY` | Alpaca paper trading | For paper mode |
| `ALPACA_PAPER_API_SECRET` | Alpaca authentication | For paper mode |

---

## 18. Data Schemas

### 18.1 Core Schemas

**Location:** `src/quantracore_apex/core/schemas.py`

```python
@dataclass
class OhlcvBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @property
    def range(self) -> float:
        return self.high - self.low

@dataclass
class OhlcvWindow:
    symbol: str
    timeframe: str
    bars: List[OhlcvBar]
    
    def get_hash(self) -> str:
        """Deterministic hash for reproducibility."""

@dataclass
class ApexResult:
    symbol: str
    timestamp: datetime
    window_hash: str
    quantrascore: int
    score_bucket: ScoreBucket
    regime: Regime
    risk_tier: RiskTier
    entropy_state: EntropyState
    suppression_state: SuppressionState
    drift_state: DriftState
    microtraits: Microtraits
    entropy_metrics: EntropyMetrics
    suppression_metrics: SuppressionMetrics
    drift_metrics: DriftMetrics
    continuation_metrics: ContinuationMetrics
    volume_metrics: VolumeMetrics
    protocol_results: List[ProtocolResult]
    verdict: Verdict
    omega_overrides: List[OmegaStatus]
```

### 18.2 Enums

```python
class Regime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    VOLATILE = "volatile"
    COMPRESSED = "compressed"
    UNKNOWN = "unknown"

class RiskTier(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

class EntropyState(Enum):
    STABLE = "stable"
    ELEVATED = "elevated"
    CHAOTIC = "chaotic"

class SuppressionState(Enum):
    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"

class DriftState(Enum):
    NONE = "none"
    MILD = "mild"
    SIGNIFICANT = "significant"
    CRITICAL = "critical"
```

---

## 19. Testing Framework

### 19.1 Test Organization

```
tests/
├── broker/              # Broker layer tests
├── core/                # Core engine tests
├── eeo_engine/          # EEO Engine tests
├── extreme/             # Extreme scenario tests
├── hardening/           # Hardening tests
├── integrations/        # Integration tests
├── lab/                 # ApexLab tests
├── matrix/              # Protocol matrix tests
├── model/               # ApexCore tests
├── nuclear/             # Nuclear determinism tests
├── perf/                # Performance tests
├── protocols/           # Protocol tests
├── regulatory/          # Regulatory compliance tests
└── scanner/             # Scanner tests
```

### 19.2 Test Counts

| Category | Tests | Focus |
|----------|-------|-------|
| Total Tests | 1,145+ | Full system coverage |
| Regulatory | 163+ | Compliance verification |
| Protocol | 200+ | All 145+ protocols |
| Core Engine | 150+ | Engine components |
| Broker Layer | 100+ | Execution layer |
| ApexCore/Lab | 80+ | ML pipeline |
| Integration | 50+ | External services |

### 19.3 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific category
pytest tests/regulatory/ -v

# Run with coverage
pytest tests/ --cov=src/quantracore_apex --cov-report=html

# Run fast subset
pytest tests/ -v -m "not slow"
```

---

## 20. Deployment

### 20.1 Workflow Configuration

| Workflow | Command | Port |
|----------|---------|------|
| FastAPI Backend | `uvicorn src.quantracore_apex.server.app:app --host 0.0.0.0 --port 8000` | 8000 |
| ApexDesk Frontend | `cd dashboard && npm run dev` | 5000 |
| Alpha Factory Dashboard | `cd static && python -m http.server 8080` | 8080 |

### 20.2 Production Build

```bash
# Frontend build
cd dashboard && npm run build

# Backend (Gunicorn recommended)
gunicorn --bind 0.0.0.0:8000 --workers 4 src.quantracore_apex.server.app:app
```

### 20.3 Data Portability

All learning data is portable via JSON/pkl files:

```
data/
├── apexlab/
│   ├── feedback_samples.json    # Feedback loop data (107+ samples)
│   └── chaos_simulation_samples.json  # Simulator data (135+ samples)
└── training/
    └── models/
        ├── apexcore_demo.pkl    # Trained model
        └── apexcore_demo_metadata.json
```

---

## 21. Autonomous Trading System

### 21.1 Overview

The Autonomous Trading System provides institutional-grade automated trading capabilities with full safety controls and self-learning integration.

**Key Components:**
- **TradingOrchestrator**: Main autonomous loop coordinating all subsystems
- **SignalQualityFilter**: Institutional-grade signal filtering (A+/A only)
- **PositionMonitor**: Real-time position tracking and management
- **TradeOutcomeTracker**: Feedback loop integration for self-learning
- **PolygonWebSocketStream**: Real-time data streaming

### 21.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Autonomous Trading System                         │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  TradingOrchestrator                         │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │   │
│  │  │Scan Loop │ │Position  │ │Stream    │ │Shutdown  │       │   │
│  │  │          │ │Loop      │ │Task      │ │Handler   │       │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │
│  │  Signal    │  │  Position  │  │  Trade     │  │  Rolling   │   │
│  │  Quality   │  │  Monitor   │  │  Outcome   │  │  Window    │   │
│  │  Filter    │  │            │  │  Tracker   │  │  Manager   │   │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘   │
│                              │                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Real-Time Streaming                       │   │
│  │  ┌──────────────────┐  ┌──────────────────┐                 │   │
│  │  │ PolygonWebSocket │  │ SimulatedStream  │                 │   │
│  │  │ (Live Data)      │  │ (Testing)        │                 │   │
│  │  └──────────────────┘  └──────────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 21.3 SignalQualityFilter

Applies institutional-grade filtering to ensure only the highest quality signals are traded.

**Filter Criteria:**

| Criterion | Default Threshold | Description |
|-----------|------------------|-------------|
| QuantraScore | ≥ 75 | Minimum composite score |
| Quality Tier | A+ or A | Only top-tier signals |
| Risk Tier | ≤ high | Block extreme risk |
| Liquidity | ≥ medium | Minimum liquidity band |
| Avoid Probability | < 0.3 | Maximum avoid flag |
| Max Positions | 5 | Concurrent position limit |
| Omega Status | No LOCKED/ENFORCED | Omega directive compliance |

**Rejection Reasons:**
- `quantrascore_below_threshold`
- `quality_tier_not_a_or_a_plus`
- `risk_tier_extreme`
- `omega_directive_blocked`
- `liquidity_below_minimum`
- `max_positions_reached`
- `cooldown_active`

### 21.4 PositionMonitor

Real-time tracking and management of open positions.

**Capabilities:**
- Stop-loss monitoring and execution
- Profit target monitoring (T1, T2)
- Trailing stop adjustment
- Time-based exit enforcement
- Broker position synchronization
- Maximum adverse/favorable excursion tracking

**Exit Reasons:**
| Exit Type | Description |
|-----------|-------------|
| `stop_loss` | Protective stop hit |
| `profit_target_1` | First target reached |
| `profit_target_2` | Second target reached |
| `trailing_stop` | Trailing stop triggered |
| `time_exit` | Max bars exceeded |
| `omega_override` | Safety directive triggered |
| `eod_exit` | End of day close |

### 21.5 TradeOutcomeTracker

Records trade outcomes and feeds back to ApexLab for model improvement.

**Tracked Metrics:**
- Entry/exit prices and times
- Realized P&L (absolute and percentage)
- Maximum favorable/adverse excursion
- Bars held
- Exit reason classification
- Original signal quality metrics

**Performance Analysis:**
- Win rate by quality tier
- P&L by QuantraScore range
- Exit reason distribution
- Profit factor calculation
- Expectancy analysis

### 21.6 Real-Time Streaming

**PolygonWebSocketStream:**
- WebSocket connection to Polygon.io
- Automatic reconnection with exponential backoff
- Heartbeat monitoring
- Trade, quote, and bar message handling
- Symbol subscription management

**RollingWindowManager:**
- Maintains 100-bar rolling windows per symbol
- Aggregates tick data into OHLCV bars
- Signals when windows are ready for analysis
- Supports historical data bootstrapping

### 21.7 Configuration

```python
OrchestratorConfig(
    watchlist=["AAPL", "NVDA", "TSLA", ...],  # Symbols to monitor
    max_concurrent_positions=5,                # Position limit
    max_portfolio_exposure=0.5,                # Max 50% exposure
    max_single_position_pct=0.15,              # Max 15% per position
    scan_interval_seconds=60.0,                # Scan frequency
    quality_thresholds=QualityThresholds(
        min_quantrascore=75.0,
        required_quality_tiers=["A+", "A"],
    ),
    paper_mode=True,                           # Paper trading only
)
```

### 21.8 Running the Orchestrator

```bash
# Demo mode (60 seconds, simulated data)
python scripts/run_autonomous.py --demo

# Paper trading mode
python scripts/run_autonomous.py --mode paper --symbols AAPL,NVDA,TSLA

# Research mode (log only, no execution)
python scripts/run_autonomous.py --mode research --duration 3600

# With live Polygon stream (requires API key)
python scripts/run_autonomous.py --live-stream --mode paper
```

### 21.9 Safety Features

| Feature | Description |
|---------|-------------|
| **Mode Enforcement** | PAPER mode by default; LIVE disabled |
| **Omega Compliance** | All 20 Omega directives enforced |
| **Position Limits** | Configurable max concurrent positions |
| **Exposure Limits** | Maximum portfolio exposure cap |
| **Kill Switch** | Emergency shutdown capability |
| **Symbol Cooldowns** | Prevent rapid re-entry after exit |
| **Market Hours** | Optional respect for trading hours |

### 21.10 Directory Structure

```
src/quantracore_apex/autonomous/
├── __init__.py                    # Package exports
├── models.py                      # Data models and schemas
├── signal_quality_filter.py       # Quality filtering logic
├── position_monitor.py            # Position tracking
├── trade_outcome_tracker.py       # Outcome recording
├── trading_orchestrator.py        # Main orchestrator
└── realtime/
    ├── __init__.py
    ├── polygon_stream.py          # WebSocket streaming
    └── rolling_window.py          # Bar aggregation
```

---

## Appendix A: Version History

| Version | Date | Changes |
|---------|------|---------|
| 9.0-A | 2025-11-29 | Initial release with full protocol documentation, Google Docs pipeline |
| 9.0-A | 2025-11-29 | Added Autonomous Trading System with TradingOrchestrator, SignalQualityFilter, PositionMonitor, TradeOutcomeTracker |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **ApexCore** | On-device neural assistant models (scikit-learn Gradient Boosting) |
| **ApexLab** | Training pipeline and dataset builder for supervised learning |
| **EEO** | Entry/Exit Optimization Engine for trade planning |
| **Microtraits** | 15+ structural features extracted from OHLCV data |
| **MonsterRunner** | Explosive move detection protocols (MR01-MR20) |
| **Omega Directive** | Safety override protocol (Ω1-Ω20) |
| **QuantraScore** | 0-100 composite probability score from ApexEngine |
| **Tier Protocol** | Technical analysis protocol (T01-T80) |
| **Learning Protocol** | Label generation protocol for ML training (LP01-LP25) |

---

**Document Classification:** Build Specification  
**Maintainer:** QuantraCore Development Team  
**Update Policy:** Update on every system change
