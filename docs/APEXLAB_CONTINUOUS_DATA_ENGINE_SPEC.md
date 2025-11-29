# ApexLab Continuous Data Engine Specification

**Version:** v9.x — High-Volume Learning Loop  
**Author:** Jesse J. Lamont — Lamont Labs  
**Status:** Roadmap Specification

---

## Purpose

ApexLab runs as a continuous, offline data engine that:

1. Continuously pulls OHLCV from multiple APIs
2. Renders standardized chart images (multi-timeframe)
3. Feeds OHLCV + images into Apex engine in teacher mode
4. Generates deterministic labels (future outcomes, quality tiers, runner flags)
5. Stores everything as training-ready datasets
6. Triggers periodic training/evaluation/promotion of ApexCore + ApexVision models

---

## Table of Contents

1. [Objectives](#1-objectives)
2. [Process Topology](#2-process-topology)
3. [Multi-API Market Data Ingest](#3-multi-api-market-data-ingest)
4. [Chart Rendering at Scale](#4-chart-rendering-at-scale)
5. [Event Labeling](#5-event-labeling)
6. [Dataset Aggregation & Sharding](#6-dataset-aggregation--sharding)
7. [Training Launcher](#7-training-launcher)
8. [Evaluation & Promotion](#8-evaluation--promotion)
9. [Resource Management](#9-resource-management)
10. [Safety, Auditability & Compliance](#10-safety-auditability--compliance)

---

## 1. Objectives

| Goal | Description |
|------|-------------|
| High Volume | Tens of thousands of labeled events + images per day or week |
| Multi-API | Unified ingestion from Polygon, AlphaVantage, TwelveData, broker feeds |
| Rate Control | Honor rate limits, cost limits, and hardware constraints |
| Determinism | Maintain deterministic labeling and fully auditable data lineage |
| Model Evolution | Feed scheduled model upgrades (weekly/monthly) with fresh, diverse data |

---

## 2. Process Topology

### Components

| Component | Role |
|-----------|------|
| **ApexLabScheduler** | Central orchestrator; kicks off periodic jobs |
| **MarketDataIngestor** | Pulls OHLCV from multiple APIs via ApexFeed |
| **ChartRendererBatcher** | Renders OHLCV windows into standardized chart images |
| **EventLabeler** | Runs Apex engine in teacher mode to generate labels |
| **DatasetAggregator** | Builds Parquet/Arrow training shards |
| **TrainingLauncher** | Triggers ApexCore/ApexVision training runs on a schedule |
| **EvaluationAndPromotion** | Evaluates models and promotes only if metrics improve |

### Scheduling

| Schedule Type | Jobs |
|---------------|------|
| **Realtime Loop** | MarketDataIngestor runs every X minutes to append fresh OHLCV windows |
| **Daily Jobs** | ChartRendererBatcher renders images for yesterday's data |
| | EventLabeler computes future-based labels for completed outcome windows |
| | DatasetAggregator rolls up daily shards |
| **Weekly Jobs** | TrainingLauncher runs ApexCore/ApexVision training |
| | EvaluationAndPromotion decides on model promotion |

All intervals are configurable (5min, 15min, hourly, daily, weekly).

---

## 3. Multi-API Market Data Ingest

### Component: MarketDataIngestor

Uses the **ApexFeed abstraction** layer.

### Supported APIs

| Provider | Type |
|----------|------|
| Polygon | Primary |
| AlphaVantage | Secondary |
| TwelveData | Secondary |
| Tradier | Broker |
| IBKR | Broker feed |
| Alpaca | Broker feed |

### Symbol Universe

| Strategy | Description |
|----------|-------------|
| Base Universe | Full U.S. equities (mega → penny) |
| Dollar Volume | Top N by dollar volume per sector |
| Volatility Buckets | Low/mid/high volatility groupings |
| Microcap/Momentum | Runner-candidate subsets |
| Adaptive Focusing | More samples where structure is richer |

### Event Volume Targets

| Tier | Events per Day |
|------|----------------|
| Low | 10,000 |
| Mid | 50,000 |
| High | 100,000+ (institutional) |

### Ingest Strategies

- Pull OHLCV in time-batches for many symbols at once
- Use asynchronous HTTP/WebSockets where available
- Cache repeated history; only append new bars

### Rate Limit & Cost Control

| Control | Behavior |
|---------|----------|
| Per-API Budget | Request limits per minute/hour/day |
| Global Cost Budget | API units / dollar ceiling |
| Backoff | Graceful queueing under rate limits |
| Throttling | Reduce symbol count per run based on limits |
| Fallback | Prefer Polygon, fall back to others |

---

## 4. Chart Rendering at Scale

### Component: ChartRendererBatcher

### Goals

- Render tens of thousands of chart images per day or week
- Support multiple timeframes per event
- Normalize visuals for vision training

### Per-Event Outputs

| Timeframe | Examples |
|-----------|----------|
| Short-term | 1m, 5m, 15m |
| Medium-term | 1h |
| Long-term | 1d |

### Output Formats

| Format | Purpose |
|--------|---------|
| PNG | Archival/audit |
| float32 tensor | Training |
| int8 tensor | Quantized training |

### Rendering Settings

| Setting | Value |
|---------|-------|
| Canvas Size | 512 × 512 |
| Style | Clean monochrome candlesticks |
| Volume | Optional overlay |
| Grid | Minimal |
| Normalization | Per-window price axis, normalized colors |

### Performance

| Aspect | Specification |
|--------|---------------|
| Parallelism | Batch rendering with multiprocessing/async workers |
| GPU | Optional GPU-accelerated rendering |
| Capacity | 10k–100k images/day depending on hardware |
| Retention | Rolling policies (PNGs 30–90 days, tensors longer) |

---

## 5. Event Labeling

### Component: EventLabeler

### Inputs

| Input | Source |
|-------|--------|
| OHLCV windows | MarketDataIngestor |
| Chart images | ChartRendererBatcher |
| Engine snapshot | Apex deterministic teacher |

### Process

1. Identify structural events (protocol triggers, regime transitions, runner candidates)
2. Run Apex engine in teacher mode at event_time
3. Wait for outcome windows (1d/3d/5d/10d) to elapse
4. Compute future outcome metrics: `ret_*`, `max_runup_*`, `max_drawdown_*`
5. Assign quality tiers (A+/A/B/C/D)
6. Assign runner / monster_runner flags
7. Assign avoid_trade flags where deterministic rules apply
8. Attach visual features/metadata if images available

### Outputs

- ApexLab v2 event rows with both OHLCV + image references
- Stored in Parquet/Arrow shards partitioned by date and symbol

---

## 6. Dataset Aggregation & Sharding

### Component: DatasetAggregator

### Behavior

- Roll up daily shards into larger training sets
- Partition by date, regime, sector
- Maintain metadata: symbol counts, regime distribution, label distribution

### Storage Layout

```
data/apexlab_v2/
├── by_date/
│   ├── 2025-01-01/
│   ├── 2025-01-02/
│   └── ...
└── by_regime/
    ├── trend_up/
    ├── trend_down/
    ├── chop/
    ├── squeeze/
    └── crash/
```

### Shard Sizes

Configurable: 10k–50k events per shard

### Lineage Metadata

| Field | Description |
|-------|-------------|
| `engine_snapshot_id` | Apex engine version |
| `scanner_snapshot_id` | Scanner version |
| `vision_model_snapshot_id` | Vision model version (if applicable) |
| `shard_hash` | Content hash |

**Guarantee:** Every dataset is reproducible from engine + feed snapshots.

---

## 7. Training Launcher

### Component: TrainingLauncher

### Schedule Modes

| Mode | Frequency | Description |
|------|-----------|-------------|
| Conservative | Monthly | Maximum stability; minimal model churn |
| Balanced | Weekly/Bi-weekly | Continuous improvement; institutional stable |
| Aggressive Research | Daily/Weekly | Lab experiments only, not for live ranking |

### Functions

1. Select appropriate dataset snapshots
2. Kick off ApexCore v2 training runs (Big + Mini)
3. Kick off ApexVision / Fusion training runs when vision data available
4. Log all hyperparameters, versions, and metrics

### Outputs

| Output | Location |
|--------|----------|
| Models | `models/apexcore_v2/*`, `models/apexvision_v1/*` |
| Manifests | Model metrics + thresholds |
| Logs | Training logs and eval reports |

---

## 8. Evaluation & Promotion

### Component: EvaluationAndPromotion

### Required Metrics

| Metric | Description |
|--------|-------------|
| Runner AUC | ApexCore & VisionFusion |
| Brier Score | Calibration for runner_prob |
| QuantraScore Alignment | MSE/MAE vs teacher |
| Quality Tier Accuracy | Tier prediction accuracy |
| Regime Performance | No hidden regime failures |
| Visual Uncertainty | Curves for vision models |

### Promotion Policy

| Rule | Description |
|------|-------------|
| Core Metrics | New model must meet or exceed prior on runner AUC, calibration |
| No Regression | No catastrophic regression in any major regime |
| Rejection | If any critical metric fails, model is rejected |

### Promotion Modes

| Mode | Description |
|------|-------------|
| Manual | Default; human/agent reviews eval report and flips pointer |
| Auto | For institutions; if all thresholds pass, manifest auto-updates |

### Active Manifest Paths

```
models/apexcore_v2/big/active_manifest.json
models/apexcore_v2/mini/active_manifest.json
models/apexvision/active_manifest.json
```

### Fail-Closed

- PredictiveAdvisor and ApexVision strict-load only from active manifests
- If manifest invalid or missing → predictive/vision disabled, engine only

---

## 9. Resource Management

### CPU/GPU

| Behavior | Description |
|----------|-------------|
| Auto-detect | Detect hardware (K6 vs institutional server) |
| Adapt | Adjust batch sizes, workers, symbol counts |

### Storage Policies

| Data Type | Retention |
|-----------|-----------|
| Raw PNGs | 30 days |
| Tensors | 180 days |
| Datasets | 365+ days (or per-institution policy) |
| Pruning | Older images compressed/pruned after training |

### API Cost Control

| Setting | Description |
|---------|-------------|
| Daily Limit | Global daily API unit limit |
| Alerts | Thresholds when hitting % of quota |
| Fallback | Automatic smaller universes under tight budget |

---

## 10. Safety, Auditability & Compliance

### Principles

1. All learning remains offline and controlled
2. No direct online-learning loop mutates engine behavior on-the-fly
3. Every dataset and model version has clear provenance (hashes, manifests)
4. Predictive outputs (ApexCore/ApexVision) are rankers only, not raw trade commands

### Audit Trail Logs

| Log | Contents |
|-----|----------|
| `data_ingest_log` | APIs called, symbols, bar counts, rate limiting |
| `render_log` | Images rendered, timeframes, failures |
| `label_log` | Events labeled, outcome windows used |
| `training_log` | Hyperparams, dataset IDs, metrics |
| `promotion_log` | Reasons for promotion/rejection of models |

### Guarantees

- Ability to reconstruct: which data shaped which model, and why promotion occurred
- Full regulatory compliance with audit requirements
- Deterministic reproducibility of all training runs

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ApexLabScheduler                              │
│                    (Central Orchestrator)                            │
└─────────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ MarketData    │      │ ChartRenderer │      │ EventLabeler  │
│ Ingestor      │─────▶│ Batcher       │─────▶│ (Teacher Mode)│
└───────────────┘      └───────────────┘      └───────────────┘
        │                       │                       │
        │                       │                       ▼
        │                       │              ┌───────────────┐
        │                       │              │ Dataset       │
        │                       └─────────────▶│ Aggregator    │
        │                                      └───────────────┘
        │                                              │
        │                                              ▼
        │                                      ┌───────────────┐
        │                                      │ Training      │
        │                                      │ Launcher      │
        │                                      └───────────────┘
        │                                              │
        │                                              ▼
        │                                      ┌───────────────┐
        └─────────────────────────────────────▶│ Evaluation &  │
                                               │ Promotion     │
                                               └───────────────┘
```

---

## Data Flow Summary

| Phase | Input | Output |
|-------|-------|--------|
| Ingest | API feeds | OHLCV windows |
| Render | OHLCV windows | Chart images + tensors |
| Label | OHLCV + images | Event rows with labels |
| Aggregate | Daily events | Parquet shards |
| Train | Dataset shards | New models |
| Evaluate | New models | Metrics report |
| Promote | Passing models | Active manifests |

---

*ApexLab Continuous Data Engine Specification | QuantraCore Apex v9.x | Lamont Labs*
