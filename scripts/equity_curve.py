#!/usr/bin/env python3
"""
EQUITY CURVE SIMULATOR (Fast Mode)
===================================
Simulates trading performance based on QuantraScore decisions.
Uses synthetic returns for demonstration when API is rate-limited.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

proof_file = "proof_logs/real_world_proof.jsonl"
if not os.path.exists(proof_file):
    print(f"ERROR: {proof_file} not found. Run force_the_truth.py first.")
    sys.exit(1)

with open(proof_file) as f:
    decisions = [json.loads(line) for line in f]

print("=" * 60)
print("QUANTRACORE APEX — EQUITY CURVE SIMULATION")
print("=" * 60)
print(f"\nLoaded {len(decisions)} decisions from proof log")

portfolio_value = 100000.0
starting_capital = portfolio_value

HIGH_THRESHOLD = 60
LOW_THRESHOLD = 30
POSITION_SIZE_PCT = 0.10

np.random.seed(42)

print(f"\nStrategy Parameters:")
print(f"  Entry threshold: QuantraScore >= {HIGH_THRESHOLD}")
print(f"  Exit threshold: QuantraScore <= {LOW_THRESHOLD}")
print(f"  Position size: {POSITION_SIZE_PCT*100:.0f}% of portfolio per trade")
print(f"  Starting capital: ${starting_capital:,.2f}")

positions = {}
trades = []
trade_log = []

print(f"\nProcessing {len(decisions)} symbols...")
print("-" * 60)

for decision in decisions:
    symbol = decision["symbol"]
    score = decision["quantra_score"]
    regime = decision.get("regime", "unknown")
    risk_tier = decision.get("risk_tier", "unknown")
    
    if regime == "trending_up":
        base_return = np.random.uniform(0.05, 0.25)
    elif regime == "trending_down":
        base_return = np.random.uniform(-0.20, -0.05)
    elif regime == "volatile":
        base_return = np.random.uniform(-0.15, 0.20)
    elif regime == "range_bound":
        base_return = np.random.uniform(-0.05, 0.10)
    else:
        base_return = np.random.uniform(-0.10, 0.15)
    
    if risk_tier == "low":
        volatility_mult = 0.5
    elif risk_tier == "medium":
        volatility_mult = 1.0
    else:
        volatility_mult = 1.5
    
    simulated_return = base_return * volatility_mult
    
    if score >= HIGH_THRESHOLD:
        position_value = portfolio_value * POSITION_SIZE_PCT
        
        pnl = position_value * simulated_return
        portfolio_value += pnl
        
        trade_log.append({
            "action": "LONG",
            "symbol": symbol,
            "score": score,
            "regime": regime,
            "risk_tier": risk_tier,
            "return": simulated_return,
            "pnl": pnl
        })
        print(f"  LONG {symbol}: Score={score:.1f}, Regime={regime}, Return={simulated_return*100:+.1f}%, PnL=${pnl:+,.2f}")
    
    elif score <= LOW_THRESHOLD:
        position_value = portfolio_value * POSITION_SIZE_PCT
        
        pnl = position_value * (-simulated_return)
        portfolio_value += pnl
        
        trade_log.append({
            "action": "SHORT",
            "symbol": symbol,
            "score": score,
            "regime": regime,
            "risk_tier": risk_tier,
            "return": -simulated_return,
            "pnl": pnl
        })
        print(f"  SHORT {symbol}: Score={score:.1f}, Regime={regime}, Return={-simulated_return*100:+.1f}%, PnL=${pnl:+,.2f}")
    
    else:
        print(f"  HOLD {symbol}: Score={score:.1f} (no action)")

total_return = (portfolio_value / starting_capital - 1) * 100

print("\n" + "=" * 60)
print("EQUITY CURVE SUMMARY")
print("=" * 60)

print(f"\n--- PORTFOLIO PERFORMANCE ---")
print(f"  Starting Capital: ${starting_capital:,.2f}")
print(f"  Ending Value: ${portfolio_value:,.2f}")
print(f"  Total Return: {total_return:+.2f}%")
print(f"  Absolute P&L: ${portfolio_value - starting_capital:+,.2f}")

print(f"\n--- TRADING STATISTICS ---")
print(f"  Total Signals: {len(decisions)}")
print(f"  Trades Executed: {len(trade_log)}")
print(f"  No-Action (Hold): {len(decisions) - len(trade_log)}")

if trade_log:
    longs = [t for t in trade_log if t["action"] == "LONG"]
    shorts = [t for t in trade_log if t["action"] == "SHORT"]
    print(f"  Long Trades: {len(longs)}")
    print(f"  Short Trades: {len(shorts)}")
    
    winners = [t for t in trade_log if t["pnl"] > 0]
    losers = [t for t in trade_log if t["pnl"] < 0]
    win_rate = len(winners) / len(trade_log) * 100 if trade_log else 0
    print(f"\n--- WIN/LOSS ANALYSIS ---")
    print(f"  Win Rate: {win_rate:.1f}% ({len(winners)}/{len(trade_log)})")
    
    if winners:
        avg_win = np.mean([t["pnl"] for t in winners])
        print(f"  Avg Win: ${avg_win:+,.2f}")
    if losers:
        avg_loss = np.mean([t["pnl"] for t in losers])
        print(f"  Avg Loss: ${avg_loss:+,.2f}")
    
    if winners and losers:
        profit_factor = abs(sum(t["pnl"] for t in winners) / sum(t["pnl"] for t in losers))
        print(f"  Profit Factor: {profit_factor:.2f}")

print(f"\n--- REGIME BREAKDOWN ---")
regimes = {}
for t in trade_log:
    r = t["regime"]
    if r not in regimes:
        regimes[r] = {"count": 0, "pnl": 0}
    regimes[r]["count"] += 1
    regimes[r]["pnl"] += t["pnl"]

for r, stats in sorted(regimes.items(), key=lambda x: x[1]["pnl"], reverse=True):
    print(f"  {r}: {stats['count']} trades, P&L=${stats['pnl']:+,.2f}")

print(f"\n--- BENCHMARK COMPARISON ---")
spy_return_2024 = 26.5
print(f"  SPY Return (2024 est.): {spy_return_2024:+.1f}%")
print(f"  Strategy Return: {total_return:+.2f}%")
print(f"  Alpha vs SPY: {total_return - spy_return_2024:+.2f}%")

sharpe_estimate = total_return / 15.0 if total_return > 0 else total_return / 20.0
print(f"  Est. Sharpe Ratio: {sharpe_estimate:.2f}")

print("\n" + "=" * 60)
print("COMPLIANCE NOTICE")
print("=" * 60)
print("These are SIMULATED returns for demonstration purposes only.")
print("Actual results will vary. Past performance does not guarantee future results.")
print("All outputs are structural probabilities, NOT trading advice.")
print("=" * 60)
