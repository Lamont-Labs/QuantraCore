#!/usr/bin/env python3
"""
Performance Report Generator for Investor Track Record Documentation

Generates comprehensive performance reports from paper trading data
for investor due diligence purposes.

Usage:
    python scripts/generate_performance_report.py
    python scripts/generate_performance_report.py --output investor_logs/reports/
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    print("Installing requests...")
    os.system("pip install requests")
    import requests


API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
OUTPUT_DIR = Path("investor_logs/reports")


def fetch_api(endpoint: str) -> dict | None:
    """Fetch data from API endpoint."""
    try:
        response = requests.get(f"{API_BASE}{endpoint}", timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Warning: Could not fetch {endpoint}: {e}")
        return None


def generate_portfolio_snapshot() -> dict[str, Any]:
    """Generate current portfolio snapshot."""
    data = fetch_api("/portfolio/status")
    if not data:
        return {"error": "Could not fetch portfolio data"}
    
    snapshot = data.get("snapshot", {})
    positions = data.get("positions", [])
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "summary": {
            "total_equity": snapshot.get("total_equity", 0),
            "cash": snapshot.get("cash", 0),
            "positions_value": snapshot.get("positions_value", 0),
            "total_pnl": snapshot.get("total_pnl", 0),
            "total_pnl_pct": snapshot.get("total_pnl_pct", 0),
            "num_positions": snapshot.get("num_positions", 0),
            "long_exposure": snapshot.get("long_exposure", 0),
            "short_exposure": snapshot.get("short_exposure", 0),
        },
        "positions": [
            {
                "symbol": p.get("symbol"),
                "quantity": p.get("quantity"),
                "avg_price": p.get("avg_price"),
                "market_value": p.get("market_value"),
                "unrealized_pnl": p.get("unrealized_pnl"),
                "side": p.get("side"),
            }
            for p in positions
        ]
    }


def generate_system_status() -> dict[str, Any]:
    """Generate system health status."""
    health = fetch_api("/health")
    broker = fetch_api("/broker/status")
    autotrader = fetch_api("/autotrader/status")
    predictive = fetch_api("/predictive/status")
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "health": {
            "status": health.get("status") if health else "unknown",
            "engine": health.get("engine") if health else "unknown",
            "data_layer": health.get("data_layer") if health else "unknown",
        },
        "broker": {
            "mode": broker.get("mode") if broker else "unknown",
            "adapter": broker.get("adapter") if broker else "unknown",
            "is_paper": broker.get("is_paper", True) if broker else True,
            "equity": broker.get("equity", 0) if broker else 0,
        },
        "autotrader": {
            "enabled": autotrader.get("enabled", False) if autotrader else False,
            "mode": autotrader.get("mode") if autotrader else "unknown",
            "active_positions": autotrader.get("active_positions", 0) if autotrader else 0,
        },
        "ml_model": {
            "version": predictive.get("version") if predictive else "unknown",
            "loaded": predictive.get("model_loaded", False) if predictive else False,
            "training_samples": predictive.get("training_samples", 0) if predictive else 0,
        }
    }


def generate_trading_modes() -> dict[str, Any]:
    """Generate trading modes status."""
    modes = fetch_api("/ml/trading-modes")
    if not modes:
        return {"error": "Could not fetch trading modes"}
    
    mode_data = modes.get("modes", {})
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "current_tier": mode_data.get("current_tier", "unknown"),
        "data_refresh": mode_data.get("data_refresh", "unknown"),
        "trading_types": mode_data.get("trading_types", {}),
    }


def load_auto_trades() -> list[dict]:
    """Load historical auto trades from investor logs."""
    trades_dir = Path("investor_logs/auto_trades")
    trades = []
    
    if trades_dir.exists():
        for trade_file in trades_dir.glob("auto_trade_*.json"):
            try:
                with open(trade_file, "r") as f:
                    trade_data = json.load(f)
                    trades.append(trade_data)
            except Exception as e:
                print(f"Warning: Could not load {trade_file}: {e}")
    
    return sorted(trades, key=lambda x: x.get("timestamp", ""), reverse=True)


def calculate_performance_metrics(portfolio: dict, trades: list) -> dict[str, Any]:
    """Calculate key performance metrics."""
    summary = portfolio.get("summary", {})
    
    starting_capital = 100000.0
    current_equity = summary.get("total_equity", starting_capital)
    total_pnl = summary.get("total_pnl", 0)
    
    total_return_pct = ((current_equity - starting_capital) / starting_capital) * 100
    
    winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
    losing_trades = [t for t in trades if t.get("pnl", 0) < 0]
    
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
    
    return {
        "starting_capital": starting_capital,
        "current_equity": current_equity,
        "total_pnl": total_pnl,
        "total_return_pct": total_return_pct,
        "trade_count": len(trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": win_rate,
        "positions_held": summary.get("num_positions", 0),
    }


def generate_full_report() -> dict[str, Any]:
    """Generate complete performance report."""
    print("Generating performance report...")
    
    portfolio = generate_portfolio_snapshot()
    system = generate_system_status()
    modes = generate_trading_modes()
    trades = load_auto_trades()
    metrics = calculate_performance_metrics(portfolio, trades)
    
    report = {
        "report_metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "report_type": "investor_performance_summary",
            "version": "9.0-A",
            "system": "QuantraCore Apex",
        },
        "executive_summary": {
            "status": "OPERATIONAL" if system.get("health", {}).get("status") == "operational" else "DEGRADED",
            "mode": "PAPER TRADING",
            "current_equity": metrics["current_equity"],
            "total_return_pct": metrics["total_return_pct"],
            "positions": metrics["positions_held"],
            "trade_count": metrics["trade_count"],
        },
        "portfolio_snapshot": portfolio,
        "system_status": system,
        "trading_modes": modes,
        "performance_metrics": metrics,
        "recent_trades": trades[:10],
        "disclaimers": [
            "PAPER TRADING ONLY - Not financial advice",
            "Past performance does not guarantee future results",
            "This is a research platform, not an investment advisor",
            "All outputs are structural probabilities only",
        ]
    }
    
    return report


def save_report(report: dict, output_dir: Path = OUTPUT_DIR):
    """Save report to file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"performance_report_{timestamp}.json"
    filepath = output_dir / filename
    
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Report saved to: {filepath}")
    
    latest_path = output_dir / "latest_performance_report.json"
    with open(latest_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Latest report linked to: {latest_path}")
    
    return filepath


def print_summary(report: dict):
    """Print report summary to console."""
    exec_summary = report.get("executive_summary", {})
    metrics = report.get("performance_metrics", {})
    
    print("\n" + "=" * 60)
    print("QUANTRACORE APEX - PERFORMANCE REPORT")
    print("=" * 60)
    print(f"Generated: {report['report_metadata']['generated_at']}")
    print(f"Status: {exec_summary.get('status', 'UNKNOWN')}")
    print(f"Mode: {exec_summary.get('mode', 'UNKNOWN')}")
    print("-" * 60)
    print(f"Current Equity: ${metrics.get('current_equity', 0):,.2f}")
    print(f"Total P&L: ${metrics.get('total_pnl', 0):,.2f}")
    print(f"Total Return: {metrics.get('total_return_pct', 0):.2f}%")
    print(f"Positions: {metrics.get('positions_held', 0)}")
    print(f"Trades Executed: {metrics.get('trade_count', 0)}")
    print(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
    print("=" * 60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate performance report")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                       help="Output directory for reports")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    report = generate_full_report()
    filepath = save_report(report, output_dir)
    print_summary(report)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
