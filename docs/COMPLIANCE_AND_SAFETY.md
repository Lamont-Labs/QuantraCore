# Compliance and Safety

QuantraCore Apex v9.0-A is designed for research and educational purposes only. This document outlines the compliance framework and safety measures, including the institutional hardening features introduced in v9.0-A.

---

## 0. v9.0-A Safety Enhancements

### Research-Only Safety Fence

The system mode is explicitly configured in `config/mode.yaml`:

```yaml
default_mode: "research"

modes:
  research:
    allowed_actions:
      - scans
      - replays
      - simulated_orders
    disallowed_actions:
      - live_orders
      - broker_connections
```

Any live trading connector MUST check this configuration and refuse to operate in research mode.

### Redundant Scoring Protection

Dual-path QuantraScore computation with consistency checking:
- Primary QuantraScore: Canonical v8.2 implementation
- Shadow QuantraScore: Independent recomputation
- Threshold: 5.0 point maximum allowed deviation
- Major deviations trigger investigation flags

### Drift Detection

Statistical monitoring protects against distribution shifts:
- QuantraScore distribution tracking
- Regime frequency monitoring
- Score consistency failure rate
- Automatic DRIFT_GUARDED mode on severe drift

---

## 1. Research-Only Designation

### What This System Does
- Analyzes historical and synthetic market data
- Produces structural probability scores (QuantraScore 0-100)
- Identifies market regimes and patterns
- Trains machine learning models on labeled data
- Provides a dashboard for research visualization

### What This System Does NOT Do
- Execute live trades on any brokerage
- Provide investment advice or recommendations
- Guarantee any financial returns
- Connect to live trading APIs in production
- Store or transmit personal financial information

---

## 2. Mandatory Compliance Mode

Every API response includes a `compliance_note` field:

```json
{
  "compliance_note": "Research tool only - not financial advice"
}
```

This is enforced by **Omega Directive Ω4** which is always active and cannot be disabled.

---

## 3. Order Management System (Simulation Only)

The OMS module (`broker/oms.py`) operates exclusively in simulation mode:

```python
simulation_mode = True  # Always true, cannot be changed
```

Key restrictions:
- No real order submission
- No connection to live brokerages
- Paper trading only
- Explicit "SIMULATION" labeling on all operations

---

## 4. Omega Directives - Safety Locks

| Directive | Trigger | Effect |
|-----------|---------|--------|
| Ω1 | Extreme risk tier | Hard safety lock - blocks all signals |
| Ω2 | Chaotic entropy state | Entropy override - flags uncertainty |
| Ω3 | Critical drift state | Drift override - warns of instability |
| Ω4 | Always active | Compliance mode - research-only label |
| Ω5 | Strong suppression | Signal suppression lock |

These directives cannot be bypassed in code. They are fundamental safety mechanisms.

---

## 5. Data Handling

### API Keys
- Never committed to repository
- Read from environment variables only
- Example: `POLYGON_API_KEY`, `ALPHA_VANTAGE_API_KEY`

### Synthetic Data
- Default mode uses synthetic data generation
- No external API required for basic testing
- Deterministic seed ensures reproducibility

### Proof Logging
- All analyses logged with SHA-256 hashes
- Timestamps and input data preserved
- Enables audit trail and reproducibility verification

---

## 6. No Personal Data

This system does not:
- Collect user personal information
- Store financial account details
- Track individual trading history
- Share any data externally

---

## 7. User Responsibilities

By using QuantraCore Apex, you acknowledge that:

1. **Not Financial Advice** - All outputs are educational and for research purposes only
2. **No Guarantees** - Past analysis does not predict future results
3. **Your Decisions** - Any financial decisions are solely your responsibility
4. **Paper Trading** - The OMS is simulation-only and cannot execute real trades
5. **Due Diligence** - You should conduct your own research before any investment

---

## 8. Disclaimer Text

The following disclaimer applies to all system outputs:

> This software is for educational and research purposes only. It does not constitute financial advice, investment recommendations, or trading signals. All outputs are structural probability assessments based on historical data analysis. Past performance does not guarantee future results. The user assumes all responsibility for any financial decisions. No live trading functionality is enabled in this system.

---

## 9. Code Compliance Checks

### Automated Verification
```bash
# Run compliance tests
pytest src/quantracore_apex/tests/ -k "compliance or omega"
```

### Manual Verification
1. Check all endpoints return `compliance_note`
2. Verify OMS `simulation_mode` is always `True`
3. Confirm no live brokerage API connections
4. Ensure no secrets in repository

---

## 10. Contact for Compliance Questions

**Jesse J. Lamont** - Founder, Lamont Labs  
Email: lamontlabs@proton.me  
GitHub: https://github.com/Lamont-Labs

---

**Statement of Intent:**

QuantraCore Apex is built as a research tool for understanding market structure and developing analytical skills. It is explicitly not designed for, and should not be used for, making actual investment decisions without additional professional guidance and personal due diligence.

---

*Last Updated: November 2025*
