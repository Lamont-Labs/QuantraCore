# QuantraCore Apex™ — Data Layer

**Version:** 8.0  
**Component:** Unified Data Ingestion  
**Status:** Active

---

## Overview

The Data Layer is a unified deterministic ingestion framework for all market data. It ensures reproducible datasets, strict version control, and fail-closed behavior on any data integrity issues.

---

## Principles

| Principle | Description |
|-----------|-------------|
| Deterministic Ingestion | Same fetch = same data |
| Reproducible Datasets | Hash-locked data blocks |
| Strict Version Control | All data versioned |
| Local Caching | Offline-first design |
| Zero Cloud Dependence | After initial pull |
| Immutable Storage | Never modify past data |
| Hash Every Chunk | SHA-256 verification |
| Offline-First | Local operation priority |

---

## Invariant Rules

1. Data must be identical between Apex and ApexLab
2. Data must be hash-locked before training
3. Data providers may be swapped without changing internal logic
4. No direct live-trading signals may be created from data layer
5. All raw data stored in read-only mode

---

## Market Data Providers

### Default Supported

| Provider | Features |
|----------|----------|
| Polygon.io | Historical OHLCV, Live OHLCV, Aggregates, Tick/quote (optional) |
| Alpaca Market Data | Historical OHLCV, Realtime streams |
| IEX Cloud | Intraday data, Reference data |
| Tiingo | Daily OHLCV, Fundamentals |
| Finnhub | News (text-only), Earnings calendars, Economic data |

### Optional Providers

- Intrinio
- TwelveData
- AlphaVantage
- Quandl

---

## Data Modes

| Mode | Description |
|------|-------------|
| `realtime_ohlcv` | Live price data |
| `historical_ohlcv` | Past price data |
| `intraday_aggregates` | Intraday bars |
| `daily_aggregates` | Daily bars |
| `fundamentals` | Company fundamentals |
| `economic_series` | Macro economic data |
| `text_only_news_feeds` | News (no images) |

---

## Restrictions

- No images
- No unlicensed alternative data
- No model inputs from unlicensed sources

---

## Data Pipeline

### Processing Steps

```
1. Raw Fetch → Local Write
    ↓
2. Timestamp Normalization
    ↓
3. Corporate Action Adjustment
    ↓
4. Volatility-Adjusted Scaling
    ↓
5. Z-Score Normalization (Windowed)
    ↓
6. Sector Metadata Attach
    ↓
7. Hash → Lock → Load
```

### Guarantees

- Reversible normalization pipeline
- Exact replay for backtests
- Exact replay for ApexLab training

---

## Caching System

### Cache Modes

| Mode | Purpose |
|------|---------|
| RAM Cache | Runtime scanning |
| Disk Cache | Long-term reproducible datasets |
| Training Cache | ApexLab training loops |

### Cache Rules

- Cache entries must match hashed signatures
- Cache misses rebuild deterministically
- No dynamic rewriting

### Directory Structure

```
data/
├── cache/
├── historical/
├── live/
├── fundamentals/
└── sector/
```

---

## Sector & Macro Data

### Sources

- Sector ETF OHLCV
- SPY/QQQ/sector-vol indexes
- Economic calendar
- VIX (volatility index)
- Custom macro baskets

### Enriched Features

| Feature | Description |
|---------|-------------|
| `sector_strength` | Relative sector strength |
| `sector_flow` | Sector money flow |
| `sector_volatility` | Sector volatility level |
| `macro_volatility_state` | Overall market volatility |
| `sector_phase_alignment` | Cross-sector synchronization |

### Use Cases

- Regime classifier
- Drift/entropy modifiers
- Sector-context protocols T61–T65
- MonsterRunner sector-phase detection

---

## Fundamentals Engine

**Purpose:** Provide legally safe fundamental signals for structural awareness.

### Features

- Earnings dates
- Earnings surprise history
- Revenue growth YoY
- Sector classification
- Float + shares outstanding

### Compliance

- Fundamentals NEVER used as buy/sell triggers
- Fundamentals may only inform risk/instability modeling

---

## Alternative Data Layer

### Allowed Data

- Text-based news sentiment (numeric-only)
- Economic releases
- Macro commentary (converted to structured features)

### Forbidden Data

- Images
- Social media scraping
- Any license-restricted data

### Extraction Pipeline

```
Text → NLP → Numeric Sentiment Vector
                    ↓
         news_volatility_impact
```

### Apex Features Generated

| Feature | Description |
|---------|-------------|
| `news_flow` | News activity level |
| `news_volatility_factor` | News impact on volatility |
| `macro_sentiment_impact` | Macro sentiment influence |

### Rules

- No contextual trading interpretation permitted
- No directional language emerges in models

---

## Dataset Hashing

### Requirements

- SHA-256 per dataset chunk
- Hash stored in MODEL_MANIFEST
- Hash used to validate ApexLab training
- Hash used to validate ApexCore promotion/rejection

### Deterministic Replay

```
Identical Dataset + Identical Apex Version = Identical Teacher Labels
```

Any mismatch triggers hard fail.

---

## Fail-Closed Data Protection

### Rejection Rules

| Condition | Action |
|-----------|--------|
| Provider returns invalid data | Reject |
| Timestamps mismatch | Reject |
| Volume or price is null | Reject |
| Corporate action not applied | Reject |
| Normalization fails | Abort pipeline |
| Hash mismatch | Abort ApexLab training |

### Error Output

```json
{
  "error_code": "DATA_INTEGRITY_FAILURE",
  "message": "Hash mismatch on dataset block",
  "traceback_location": "data/historical/AAPL_2025.parquet",
  "dataset_block": "2025-Q1",
  "action": "pipeline_aborted"
}
```

---

## API Compliance

### Trading Restrictions

- No trading signals derived from raw API data
- No real-time recommendation language
- No predictive outputs from data feed alone

### Privacy

- No user-identifiable data stored
- Data stored only in encrypted cache
- Data cannot be exported externally

### Security

- API keys stored using environment-level encryption
- Keys not committed to repo
- Keys never logged

---

## Related Documentation

- [Core Engine](CORE_ENGINE.md)
- [ApexLab Training](APEXLAB_TRAINING.md)
- [Architecture](ARCHITECTURE.md)
