# Real-Time Data Ingestion and Continuous Training Architecture

## Overview

QuantraCore Apex v9.0-A is designed to support real-time data ingestion and continuous model training when deployed on a desktop with the appropriate data subscription.

## Current Architecture (EOD Mode)

With free Alpaca paper trading:
- **Data Refresh**: End-of-day (EOD) via Polygon.io
- **Training**: Batch training on historical data (150+ days)
- **Best For**: Swing trades (2-10 days), position trades

## Real-Time Architecture (With $99/mo Subscription)

### Enabling Real-Time Mode

1. Subscribe to **Alpaca Algo Trader Plus** ($99/month)
2. Set environment variable: `ALPACA_REALTIME_ENABLED=true`
3. Restart the application
4. All trading types become available (day trading, scalping, intraday)

### Real-Time Data Ingestion

When real-time mode is enabled:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Alpaca SIP    │────>│  Ring Buffer     │────>│  Feature Store  │
│   WebSocket     │     │  (100 symbols)   │     │  (Parquet)      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                 │
                                 v
                        ┌──────────────────┐
                        │  Background      │
                        │  Training Jobs   │
                        └──────────────────┘
```

### Components

#### 1. WebSocket Streaming
- **Connection**: `wss://stream.data.alpaca.markets/v2/sip`
- **Data Types**: Quotes, Trades, 1-minute Bars
- **Refresh Rate**: Sub-second
- **Coverage**: 100% US market (SIP data)

#### 2. Bounded Ring Buffer
- In-memory buffer for recent ticks
- Configurable size (default: 10,000 ticks per symbol)
- Auto-eviction of old data
- Zero-copy access for feature computation

#### 3. Parquet Append Storage
- Persistent storage for training data
- Efficient columnar format
- Compression: LZ4 (fast) or ZSTD (compact)
- Partitioned by date for efficient querying

#### 4. Background Training Jobs
- Low-priority CPU threads
- Incremental learning (warm_start)
- Scheduled windows (avoid trading hours)
- Checkpoint rotation for model versioning

## Desktop Deployment Considerations

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32 GB |
| Storage | 100 GB SSD | 500 GB NVMe |
| Network | 10 Mbps | 100 Mbps |

### Training Schedule

```
Market Hours (9:30 AM - 4:00 PM ET):
  - Data ingestion: ACTIVE
  - Training: PAUSED (inference only)
  
After Hours (4:00 PM - 8:00 PM ET):
  - Data ingestion: ACTIVE (if extended hours enabled)
  - Training: LIGHT (incremental updates)
  
Overnight (8:00 PM - 4:00 AM ET):
  - Data ingestion: IDLE
  - Training: INTENSIVE (full retraining, hyperparameter tuning)
```

### Rate Limit Management

| Subscription | Rate Limit | Symbols | Strategy |
|--------------|------------|---------|----------|
| Free EOD | 5/min | Unlimited | Batch daily |
| Algo Trader Plus | 10,000/min | 100 streaming | Real-time streaming |

## Hyperspeed Learning System Integration

The Hyperspeed Learning System leverages real-time data for accelerated model maturity:

### Historical Replay Engine
- 1000x speed replay of historical data
- Simulates 1 year of market action overnight
- Uses the same feature pipeline as live data

### Battle Simulator
- 100 parallel strategy simulations per cycle
- Tests model variants against each other
- Selects best-performing configurations

### Multi-Source Data Fusion
- Polygon: Price/volume data
- Alpaca: Order flow, execution data
- Binance: Crypto correlation data
- FRED: Economic indicators

### Model Maturity Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Initial Training | Day 1-3 | Historical data bootstrap |
| Hyperspeed Calibration | Day 4-7 | 1000x replay + battle simulations |
| Live Validation | Day 8-14 | Paper trading with real signals |
| Production Ready | Week 3+ | Full confidence, continuous learning |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/ml/realtime/status` | Real-time scanner status |
| `/ml/realtime/signals` | Live signals (WebSocket when enabled) |
| `/ml/trading-modes` | Available trading types by subscription |
| `/hyperspeed/status` | Hyperspeed learning progress |
| `/apexlab/continuous/status` | Continuous learning metrics |

## Implementation Details

### Continuous Training Loop

```python
while training_enabled:
    # 1. Collect new data from ring buffer
    new_samples = ring_buffer.get_recent(window=1000)
    
    # 2. Compute features
    features = feature_pipeline.transform(new_samples)
    
    # 3. Append to training dataset
    training_store.append(features)
    
    # 4. Check if training window reached
    if training_store.size >= MIN_TRAINING_SAMPLES:
        # 5. Incremental model update
        model.partial_fit(features, labels)
        
        # 6. Validate on holdout
        if validation_score > current_best:
            # 7. Hot swap to new model
            model_manager.deploy(model)
            
    # 8. Sleep until next window
    await asyncio.sleep(training_interval)
```

### Circuit Breakers

The system includes circuit breakers for API resilience:

| Provider | Failure Threshold | Reset Timeout |
|----------|------------------|---------------|
| Alpaca | 5 failures | 30 seconds |
| Polygon | 5 failures | 30 seconds |
| FRED | 3 failures | 120 seconds |
| Finnhub | 3 failures | 120 seconds |

## Summary

**Yes, QuantraCore Apex can ingest real-time data and train constantly on a desktop** when configured with:

1. **Alpaca Algo Trader Plus** ($99/month) for real-time streaming
2. **Sufficient hardware** (8+ cores, 32GB RAM recommended)
3. **ALPACA_REALTIME_ENABLED=true** environment variable

The system is architected for this use case with:
- Bounded ring buffers for memory management
- Parquet storage for efficient data persistence
- Background training with time-sliced windows
- Model shadow mode for safe deployments
- Circuit breakers for API resilience
- Checkpoint rotation for model versioning

The Hyperspeed Learning System can accelerate model maturity from months to 1-2 weeks through historical replay and parallel simulations.
