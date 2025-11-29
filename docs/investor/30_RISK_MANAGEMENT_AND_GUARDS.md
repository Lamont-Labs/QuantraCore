# Risk Management and Guards

**Document Classification:** Investor Due Diligence — Risk/Compliance  
**Version:** 9.0-A  
**Date:** November 2025  

---

## What Stops This From Taking Insane Risk?

QuantraCore Apex is designed with multiple layers of risk controls. The system defaults to research-only mode with execution disabled, and even if execution were enabled, multiple guardrails prevent excessive risk.

---

## Risk Control Architecture

```
Signal Generation → Risk Module → Omega Directives → Output
                        ↓
               ┌───────────────────┐
               │ Risk Constraints  │
               │ • max_notional    │
               │ • max_daily_loss  │
               │ • sector_exposure │
               │ • kill_switch     │
               └───────────────────┘
```

---

## Risk Constraints in Engine

### Position-Level Limits

| Constraint | Default | Purpose |
|------------|---------|---------|
| `max_notional` | $0 (disabled) | Maximum position size |
| `max_position_pct` | 0% | Maximum portfolio allocation |
| `min_liquidity` | None | Minimum daily volume |
| `max_spread_pct` | None | Maximum bid-ask spread |

### Portfolio-Level Limits

| Constraint | Default | Purpose |
|------------|---------|---------|
| `max_positions` | 0 | Maximum simultaneous positions |
| `max_sector_exposure` | 0% | Maximum per-sector allocation |
| `max_gross_exposure` | 0% | Maximum total exposure |
| `max_beta_exposure` | 0 | Maximum market beta |

### Time-Based Limits

| Constraint | Default | Purpose |
|------------|---------|---------|
| `max_daily_loss` | $0 | Daily loss limit |
| `max_weekly_loss` | $0 | Weekly loss limit |
| `max_drawdown` | 0% | Maximum peak-to-trough |
| `cooling_period_days` | N/A | Lockout after limit hit |

---

## Omega Directive Kill-Switch (Ω2)

### What Is Ω2?

Ω2 (Entropy Override) is an automatic kill-switch that halts all activity when conditions become too volatile or risky.

### Trigger Conditions

| Condition | Threshold | Action |
|-----------|-----------|--------|
| VIX spike | >40 | Halt new signals |
| Portfolio drawdown | >max_drawdown | Close all positions (if enabled) |
| Model drift detected | Drift score >0.3 | Distrust model, engine-only |
| Manual override | Operator trigger | Immediate halt |
| Integrity failure | Any manifest issue | Halt and investigate |

### Kill-Switch Behavior

```
Trigger detected → Ω2 activated → All execution halted → Alert sent → Manual review required
```

### Recovery Process

1. Trigger condition must be resolved
2. Manual review of positions and signals
3. Explicit re-enable by operator
4. Gradual ramp-up of activity

---

## Research-Only vs Execution Mode

### Research-Only (Default)

| Aspect | Behavior |
|--------|----------|
| Signal generation | Active |
| Risk assessment | Active |
| Order generation | **Disabled** |
| Position tracking | Paper only |
| Broker connection | **Disabled** |

This is the shipped configuration. The system generates research insights but cannot execute trades.

### Execution Mode (Requires Explicit Enable)

| Aspect | Behavior |
|--------|----------|
| Signal generation | Active |
| Risk assessment | Active + enforced |
| Order generation | **Enabled** |
| Position tracking | Real |
| Broker connection | **Enabled** |

This mode requires:
- Explicit configuration changes
- Non-zero risk limits
- Broker API credentials
- Operator acknowledgment

---

## Example Kill-Switch Event

### Scenario: VIX Spike

```
T+0: VIX crosses 40 threshold
T+0: Ω2 triggered automatically
T+0: All pending signals suppressed
T+0: Alert sent to operator
T+1min: No new activity permitted
T+??: Operator reviews situation
T+??: VIX returns to normal
T+??: Operator manually re-enables
T+??: Gradual activity resumption
```

### Scenario: Portfolio Drawdown

```
T+0: Portfolio down 5% from peak
T+0: Warning logged
T+1: Portfolio down 8% from peak
T+1: Ω2 triggered
T+1: If execution enabled, positions closed
T+1: Cooling period begins
T+??: Manual review required before re-enable
```

---

## Risk Module Pseudocode

```python
class RiskModule:
    def check_position_risk(self, signal):
        if signal.notional > self.config.max_notional:
            return RiskVeto("NOTIONAL_EXCEEDED")
        
        if signal.pct_of_portfolio > self.config.max_position_pct:
            return RiskVeto("POSITION_PCT_EXCEEDED")
        
        if self.get_sector_exposure(signal.sector) > self.config.max_sector:
            return RiskVeto("SECTOR_EXCEEDED")
        
        return RiskApproved()
    
    def check_portfolio_risk(self):
        if self.daily_loss > self.config.max_daily_loss:
            return TriggerKillSwitch("DAILY_LOSS_EXCEEDED")
        
        if self.drawdown > self.config.max_drawdown:
            return TriggerKillSwitch("DRAWDOWN_EXCEEDED")
        
        return PortfolioOK()
    
    def check_market_conditions(self):
        if self.vix > 40:
            return TriggerKillSwitch("VIX_SPIKE")
        
        if self.model_drift_score > 0.3:
            return TriggerDriftOverride()
        
        return MarketOK()
```

---

## Fail-Closed Philosophy

### Principle

> When in doubt, do nothing. When something fails, stop everything.

### Implementation

| Situation | Response |
|-----------|----------|
| Risk check fails | Block signal |
| Model unavailable | Use engine-only |
| Limit exceeded | Halt activity |
| Integrity failure | Stop and alert |
| Unknown error | Stop and alert |

### Never

- Fail open (proceed despite issues)
- Assume default values for limits
- Skip risk checks for speed
- Override kill-switch automatically

---

## Monitoring and Alerting

### What Gets Monitored

| Metric | Frequency | Alert Threshold |
|--------|-----------|-----------------|
| VIX level | Real-time | >35 warning, >40 halt |
| Portfolio P&L | Real-time | Loss >2% warning |
| Model predictions | Per-signal | Disagreement >0.3 |
| System health | 1-minute | Any component down |

### Alert Channels

| Severity | Channel |
|----------|---------|
| Info | Log file |
| Warning | Log + dashboard |
| Critical | Log + dashboard + operator notification |
| Emergency | All + Ω2 trigger |

---

*QuantraCore Apex v9.0-A | Lamont Labs | November 2025*
