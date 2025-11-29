# ApexVision Multi-Modal Vision Upgrade Specification

**Version:** v9.x — Apex Predictive Build + Vision Architecture Add-On  
**Author:** Jesse J. Lamont — Lamont Labs  
**Status:** Roadmap Specification

---

## Purpose

This document extends the Apex Predictive Build with a **full visual intelligence stack**:

- ApexVision (image-based structural intelligence)
- ApexLab_Vision (chart-image label factory)
- ApexCore_VisionFusion (multi-modal fusion layer)
- ApexVision Android Copilot v5.x upgrade
- Institutional Vision Dataset Pipeline
- Vision-Based Runner Detection & Structural Patterning
- Full integration into PredictiveAdvisor (fail-closed)

---

## Table of Contents

1. [New Components](#1-new-components)
2. [ApexVision — Visual Intelligence Engine](#2-apexvision--visual-intelligence-engine)
3. [Vision Pattern Dictionary](#3-vision-pattern-dictionary)
4. [ApexLab_Vision — Chart Image Label Factory](#4-apexlab_vision--chart-image-label-factory)
5. [ApexVision Dataset Pipeline](#5-apexvision-dataset-pipeline)
6. [ApexCore_VisionFusion — Multi-Modal Model](#6-apexcore_visionfusion--multi-modal-model)
7. [PredictiveAdvisor Integration](#7-predictiveadvisor-integration)
8. [Runner Detection Upgrade](#8-runner-detection-upgrade)
9. [QuantraVision Apex v5 — Android Copilot](#9-quantravision-apex-v5--android-copilot)
10. [Training Cycle & Model Evolution](#10-training-cycle--model-evolution)
11. [End-to-End Pipeline Summary](#11-end-to-end-pipeline-summary)

---

## 1. New Components

| Component | Description |
|-----------|-------------|
| **ApexVision** | Visual Pattern Intelligence Engine |
| **ApexLab_Vision** | Chart-Image Label Generator |
| **ApexCore_VisionFusion** | Multi-Modal Neural Model |
| **ApexVision Preprocessing Pipeline** | Uniform chart tensor builder |
| **ApexVision Android Overlay v5** | Real-time structural copilot |
| **Vision Pattern Dictionary** | Deterministic pattern taxonomy |
| **ApexVision Dataset Pipeline** | Licensed chart image ingestion |
| **Vision Safety Layer** | Fail-closed vision uncertainty gating |

---

## 2. ApexVision — Visual Intelligence Engine

### Description

A neural engine that interprets market structure visually by analyzing chart images. It provides deep geometric, spatial, and pattern-contextual signals that raw OHLCV cannot reveal. ApexVision becomes a **third intelligence pillar** next to Apex and ApexCore.

### Goals

- Identify patterns humans see but bars cannot encode
- Provide structural geometry signals: sweeps, wicks, compression, fractures
- Enhance runner detection with visual confirmation
- Enhance QuantraVision Android overlay with real-time structural interpretation

### Inputs

| Input | Description |
|-------|-------------|
| Chart images | Rendered from OHLCV or captured from Android screens |
| Multi-timeframe crops | 1min, 5min, 15min, 1h, 1d |
| Metadata | From Apex engine for fusion |

### Outputs

| Output | Description |
|--------|-------------|
| `visual_pattern_logits` | List of pattern classes |
| `wick_geometry_vector` | Wick shape analysis |
| `momentum_shape_vector` | Momentum visualization |
| `compression/expansion flags` | Range state |
| `sweep_signature_score` | Liquidity sweep detection |
| `pullback_symmetry_score` | Pullback quality |
| `visual_runner_score` | Runner probability from vision |
| `visual_uncertainty` | Confidence measure |

### Fail-Closed Rules

1. If `visual_uncertainty > threshold` → disable visual outputs
2. If visual disagrees with engine in volatile regimes → engine wins
3. If manifest hash mismatch → vision disabled

---

## 3. Vision Pattern Dictionary

A deterministic set of visually-defined chart patterns ApexVision can detect.

### Wick Patterns

| Pattern | Description |
|---------|-------------|
| `long_upper_rejection` | Strong selling pressure |
| `long_lower_rejection` | Strong buying pressure |
| `wick_compression` | Tightening range |
| `wick_expansion` | Increasing volatility |

### Body Patterns

| Pattern | Description |
|---------|-------------|
| `micro_body_cluster` | Indecision/consolidation |
| `large_engulfing` | Strong directional move |
| `thin_range` | Low volatility period |
| `gap_continuation` | Gap with follow-through |

### Structural Patterns

| Pattern | Description |
|---------|-------------|
| `rounded_bottom` | Accumulation pattern |
| `rounded_top` | Distribution pattern |
| `cup_shape` | Cup formation |
| `flag_shape` | Continuation flag |
| `wedge_shape` | Converging trendlines |
| `ascending_channel` | Upward channel |
| `descending_channel` | Downward channel |
| `sweep_structure` | Liquidity sweep |
| `liquidity_grab_signature` | Stop hunt pattern |

### Multi-Timeframe Patterns

| Pattern | Description |
|---------|-------------|
| `frame_overlap` | Cross-timeframe alignment |
| `multi_frame_confluence` | Multiple timeframe agreement |
| `aligned_runner_ramp` | Runner across timeframes |
| `compressed_range_break` | Breakout from compression |

### Runner-Specific Patterns

| Pattern | Description |
|---------|-------------|
| `runner_precoil` | Pre-breakout compression |
| `runner_compression_cluster` | Tight range before move |
| `monster_runner_visual_signature` | Extreme move setup |

---

## 4. ApexLab_Vision — Chart Image Label Factory

### Description

Deterministic labeling system for chart images. Takes raw OHLCV, renders chart images, extracts structural visual features, and attaches deterministic future outcome labels.

### Chart Renderer

Converts OHLCV sequences into standardized chart images.

| Setting | Value |
|---------|-------|
| Canvas Size | 512 × 512 |
| Color Scheme | Monochrome candles |
| Grid | Optional |
| Overlays | Volume bars, VWAP, optional MAs |
| Output Format | PNG → float tensor |

### Deterministic Labels

| Label | Description |
|-------|-------------|
| `future_quality_tier` | A+/A/B/C/D grading |
| `future_runup_percentage` | Forward return |
| `future_drawdown_percentage` | Maximum adverse excursion |
| `max_runup_5d` | 5-day maximum gain |
| `max_drawdown_5d` | 5-day maximum loss |
| `runner_prob` | Teacher-generated probability |

### Visual Labels

| Label | Description |
|-------|-------------|
| `pattern_class` | Detected pattern type |
| `pattern_confidence` | Detection confidence |
| `symmetry_score` | Pattern symmetry |
| `sweep_signature_score` | Sweep detection |
| `momentum_shape_class` | Momentum classification |

### Dataset Structure

| Column | Type |
|--------|------|
| `chart_tensor` | float32 tensor |
| `symbol` | string |
| `timeframe` | string |
| `engine_snapshot_id` | string |
| `pattern_labels` | dict |
| `future_outcomes` | dict |
| `visual_metadata` | dict |

**File Format:** Parquet / Arrow

---

## 5. ApexVision Dataset Pipeline

### Description

Pipeline for ingesting licensed chart-images, vendor datasets, Android captures, synthetic charts, and multi-timeframe rendered images into a unified dataset.

### Stages

#### Stage 1: Raw Image Ingest

- Load licensed vendor sets
- Load synthetic rendered charts from OHLCV
- Load Android screenshots from QuantraVision logs

#### Stage 2: Image Standardization

- Resize to 512×512
- Normalize candle colors
- Normalize axes
- Crop multi-frame regions

#### Stage 3: Tensor Encoding

- Convert to float32 or int8 quantized tensors
- Embed metadata

#### Stage 4: Label Attachment

- Attach deterministic ApexLab v2 future labels
- Attach pattern labels from pattern dictionary

---

## 6. ApexCore_VisionFusion — Multi-Modal Model

### Description

Fusion model that merges OHLCV-sequence embeddings, protocol vectors, and ApexVision chart-image embeddings into a unified multi-modal decision engine.

### Architecture

#### Sequence Backbone

| Property | Value |
|----------|-------|
| Type | Transformer/CNN |
| Input | 128–256 bars OHLCV |

#### Vision Backbone Options

| Model | Description |
|-------|-------------|
| ViT-tiny | Small Vision Transformer |
| ResNet-lite | Lightweight ResNet |
| MobileNetV3-small | Mobile-optimized |

**Output:** `visual_latent_vector`

#### Fusion Head

| Property | Value |
|----------|-------|
| Type | Cross-attention or concat+MLP |
| Output | `fusion_latent_vector` |

### Model Outputs

| Output | Source |
|--------|--------|
| `quantra_score` | Mandatory |
| `runner_prob` | ApexCore |
| `visual_runner_score` | ApexVision |
| `future_quality_tier_logits` | Fusion |
| `avoid_trade_prob` | Safety |
| `visual_pattern_logits` | Vision |
| `visual_uncertainty` | Confidence |

### Ensemble Configuration

| Property | Value |
|----------|-------|
| Ensemble Size | 3–7 models |
| Uncertainty Metric | ensemble_disagreement + visual_uncertainty |

---

## 7. PredictiveAdvisor Integration

### Updated Behavior

- PredictiveAdvisor now uses: OHLCV embeddings + protocol vectors + visual embeddings
- Fusion outputs used for ranking only (UPRANK / DOWNRANK / AVOID)
- Visual layer can veto an ApexCore suggestion if visual_uncertainty is high
- **Engine (Apex) STILL overrides EVERYTHING — determinism first**

### Fail-Closed Logic

| Condition | Action |
|-----------|--------|
| `visual_uncertainty > X` | Ignore visual model entirely |
| `visual_runner_score < Y` | No up-rank allowed |
| Vision manifest invalid | Revert to text-only ApexCore |

---

## 8. Runner Detection Upgrade

### New Visual Intelligence

| Feature | Description |
|---------|-------------|
| Visual pre-coil detection | Compression before breakout |
| Compression cluster boundary | Tight range identification |
| Multi-frame runner confirmation | Cross-timeframe validation |
| Runner wick symmetry | Wick pattern analysis |
| Monster runner visual signature | Extreme setup detection |

### Effects

- MonsterRunner Fuse Engine becomes significantly more accurate
- ApexCore fusion layer provides confidence scores
- Visual signals integrated as confirmation or veto, **never direct trading commands**

---

## 9. QuantraVision Apex v5 — Android Copilot

### Description

Real-time structural chart interpreter using ApexCore_VisionFusion Mini. Interprets chart screenshots on-device, overlays structural zones, wicks, sweeps, and pattern labels.

### Features

| Feature | Description |
|---------|-------------|
| Real-time capture | MediaProjection API |
| Inference speed | 30–60ms |
| Structural bounding boxes | Zone visualization |
| Wick geometry overlays | Wick analysis display |
| Pattern labels | Non-predictive structure |
| Runner visual highlights | Runner signature display |
| Fail-closed safety | Uncertainty threshold gating |
| ApexLite | Local protocol subset |

### Safety Rules

1. **No buy/short advice** — structure only
2. Fail-closed if pattern confidence < threshold

---

## 10. Training Cycle & Model Evolution

### Dataset Growth

| Property | Value |
|----------|-------|
| Schedule | Daily append (OHLCV + chart images) |
| Determinism | Dataset snapshots hashed and versioned |

### Training Schedule Options

| Frequency | Risk Level |
|-----------|------------|
| Weekly | Balanced |
| Bi-weekly | Conservative |
| Monthly | Institutional ultra-safe |

### Evaluation Requirements

| Metric | Requirement |
|--------|-------------|
| Runner AUC | >= prior model |
| Visual calibration | >= prior model |
| Uncertainty | <= allowed thresholds |
| QuantraScore alignment | Maintained |
| Pattern classification | Stable |

### Promotion Logic

| Condition | Action |
|-----------|--------|
| ALL thresholds pass | Auto-promote |
| ANY threshold fails | Reject, retain prior version |

### Manifest Contents

- Model hashes
- Vision model metrics
- Fusion metrics
- Visual uncertainty curves
- Runner fusion calibration
- Failure reasons (if rejected)

---

## 11. End-to-End Pipeline Summary

### Phase 1: Data Capture

- Apex engine captures deterministic events
- Render chart images
- Android overlay images captured

### Phase 2: Dataset Build

- OHLCV windows
- Chart tensors
- Pattern labels
- Future labels

### Phase 3: Training

- Sequence model
- Vision model
- Fusion model
- Ensemble training

### Phase 4: Evaluation

- Calibration
- Runner metrics
- Visual metrics
- Multi-frame performance

### Phase 5: Integration

- PredictiveAdvisor loads fusion ensemble
- QuantraVision Apex uses mini fusion model
- **Engine remains deterministic authority**

---

## Core Principles

1. **Determinism First** — Engine always overrides visual layer
2. **Fail-Closed Always** — Uncertainty disables visual outputs
3. **No Trading Advice** — Structure interpretation only
4. **Manifest Verification** — Hash mismatch disables vision
5. **Research Only** — All outputs are structural probabilities

---

*ApexVision Multi-Modal Upgrade Specification | QuantraCore Apex v9.x | Lamont Labs*
