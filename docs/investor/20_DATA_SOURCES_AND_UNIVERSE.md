# Data Sources and Universe

**Document Classification:** Investor Due Diligence — Data/Training  
**Version:** 9.0-A  
**Date:** November 2025  

---

## Where Does the Historical Data Come From?

QuantraCore Apex uses multiple data sources to ensure reliability and coverage. The system is designed to work with standard OHLCV (Open, High, Low, Close, Volume) data from any compatible source.

---

## Data Vendors

### Primary: Polygon.io

| Aspect | Details |
|--------|---------|
| **Type** | REST API |
| **Coverage** | US equities, full historical |
| **Frequency** | Daily, intraday available |
| **Tier Required** | Free tier for basic, paid for full history |
| **API Key** | Required (`POLYGON_API_KEY`) |

**Advantages:**
- Comprehensive historical data
- Reliable uptime
- Clean data with adjustments
- Good documentation

### Backup: Alpha Vantage

| Aspect | Details |
|--------|---------|
| **Type** | REST API |
| **Coverage** | Global equities, crypto, forex |
| **Frequency** | Daily, intraday |
| **Tier Required** | Free tier for basic |
| **API Key** | Optional (`ALPHA_VANTAGE_API_KEY`) |

**Use Case:** Backup when Polygon unavailable or for additional coverage.

### Backup: Yahoo Finance

| Aspect | Details |
|--------|---------|
| **Type** | Unofficial API (yfinance) |
| **Coverage** | Global equities, indices, ETFs |
| **Frequency** | Daily |
| **Tier Required** | Free |
| **API Key** | Not required |

**Use Case:** Development, testing, backup for basic data.

### Historical Import: CSV Bundle

| Aspect | Details |
|--------|---------|
| **Type** | Local file import |
| **Format** | CSV with standard schema |
| **Coverage** | User-defined |
| **Use Case** | Historical data import, offline operation |

### Testing: Synthetic Data

| Aspect | Details |
|--------|---------|
| **Type** | Generated data |
| **Purpose** | Unit testing, determinism verification |
| **Coverage** | Controlled scenarios |

---

## Universe Composition

### Market Cap Buckets

| Bucket | Market Cap Range | Typical Count |
|--------|-----------------|---------------|
| **Mega** | >$200B | ~50 |
| **Large** | $10B-$200B | ~300 |
| **Mid** | $2B-$10B | ~500 |
| **Small** | $300M-$2B | ~1,500 |
| **Micro** | $50M-$300M | ~3,000 |
| **Nano** | <$50M | ~5,000+ |
| **Penny** | <$5 price | Variable |

### Sector Coverage

| Sector | Included |
|--------|----------|
| Technology | Yes |
| Healthcare | Yes |
| Financial | Yes |
| Consumer | Yes |
| Industrial | Yes |
| Energy | Yes |
| Materials | Yes |
| Utilities | Yes |
| Real Estate | Yes |
| Communication | Yes |

### Liquidity Filters

| Filter | Default Threshold | Purpose |
|--------|-------------------|---------|
| Min Avg Volume | 100,000 shares | Ensure tradability |
| Min Price | $1.00 | Avoid extreme penny stocks |
| Max Spread | 2% | Ensure reasonable execution |

---

## Time Coverage

### Historical Depth

| Period | Coverage |
|--------|----------|
| **Full History** | 2000-present (Polygon paid) |
| **Standard** | 5+ years (Polygon free) |
| **Minimum** | 2 years (for model training) |

### Update Frequency

| Mode | Update Schedule |
|------|-----------------|
| **Research** | Daily EOD |
| **Lab Training** | Batch (weekly/monthly) |
| **Live Analysis** | Intraday (when data available) |

---

## Known Gaps and Limitations

### Data Quality Issues

| Issue | Frequency | Handling |
|-------|-----------|----------|
| Missing bars | Rare | Forward-fill or exclude |
| Corporate actions | Common | Use adjusted data |
| Splits not adjusted | Rare | Apply adjustment factors |
| Outlier prices | Very rare | Sanity check filters |

### Coverage Gaps

| Gap | Impact | Mitigation |
|-----|--------|------------|
| Delisted stocks | Historical only | Track delistings separately |
| OTC/Pink sheets | Limited | Focus on exchange-listed |
| International | Varies by source | Multi-source aggregation |
| After-hours | Limited | Use regular session only |

---

## Data Schema

### Standard OHLCV

| Field | Type | Description |
|-------|------|-------------|
| `symbol` | string | Ticker symbol |
| `date` | date | Trading date |
| `open` | float | Opening price |
| `high` | float | High price |
| `low` | float | Low price |
| `close` | float | Closing price |
| `volume` | int | Trading volume |
| `adjusted_close` | float | Split-adjusted close |

### Extended Fields (When Available)

| Field | Type | Description |
|-------|------|-------------|
| `vwap` | float | Volume-weighted avg price |
| `trades` | int | Number of trades |
| `market_cap` | float | Market capitalization |
| `sector` | string | GICS sector |

---

## Training Universe

### What Was Used for Current Models

| Aspect | Value |
|--------|-------|
| **Symbols** | ~3,000 liquid US equities |
| **Period** | 2018-2024 |
| **Bars** | ~2 million windows |
| **Quality Filter** | Avg volume >200k, price >$5 |

### Walk-Forward Splits

| Split | Period | Purpose |
|-------|--------|---------|
| **Train** | 2018-2022 | Model fitting |
| **Validation** | 2022-2023 | Hyperparameter tuning |
| **Test** | 2023-2024 | Final evaluation |

---

## Data Integrity

### Validation Checks

| Check | Purpose |
|-------|---------|
| Schema validation | Ensure correct field types |
| Price sanity | Detect impossible prices |
| Volume sanity | Detect impossible volumes |
| Gap detection | Identify missing bars |
| Duplicate detection | Remove duplicate records |

### Provenance Tracking

Each dataset includes:
- Source identifier
- Fetch timestamp
- Row count
- Hash of contents
- Validation status

---

*QuantraCore Apex v9.0-A | Lamont Labs | November 2025*
