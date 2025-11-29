#!/usr/bin/env python3
"""
Run Alpha Factory in Demo/Synthetic Mode.

This script runs the Alpha Factory feedback loop using synthetic data,
generating signals that feed back into the training pipeline.

Usage:
    python scripts/run_alpha_factory_demo.py [--cycles N] [--symbols SYMBOLS]
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import timedelta
from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter
from src.quantracore_apex.core.schemas import OhlcvBar

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class AlphaFactoryDemo:
    """
    Demo mode Alpha Factory that generates signals from synthetic data.
    
    Signals feed back into training pipeline via feedback samples.
    """
    
    DEFAULT_SYMBOLS = ["AAPL", "NVDA", "TSLA", "SPY", "QQQ", "AMZN", "GOOG", "META"]
    
    def __init__(
        self,
        symbols: List[str] = None,
        output_path: str = "data/apexlab/feedback_samples.json"
    ):
        self.symbols = symbols or self.DEFAULT_SYMBOLS
        self.output_path = Path(output_path)
        self.engine = ApexEngine(enable_logging=False)
        self.data_adapter = SyntheticAdapter()
        self._signals_generated = 0
    
    def run_cycle(self) -> List[Dict[str, Any]]:
        """Run a single cycle of signal generation."""
        signals = []
        
        for symbol in self.symbols:
            try:
                end = datetime.now()
                start = end - timedelta(days=200)
                bars = self.data_adapter.fetch_ohlcv(
                    symbol=symbol,
                    start=start,
                    end=end,
                    timeframe="1d"
                )
                
                if len(bars) < 100:
                    logger.warning(f"Insufficient bars for {symbol}")
                    continue
                
                window_bars = bars[-100:]
                
                apex_result = self.engine.run_scan(window_bars, symbol, seed=42)
                
                signal = self._create_signal(symbol, apex_result, window_bars)
                signals.append(signal)
                self._signals_generated += 1
                
                logger.info(f"Generated signal for {symbol}: "
                           f"score={apex_result.quantrascore:.1f}, "
                           f"risk={apex_result.risk_tier.value}, "
                           f"regime={apex_result.regime.value}")
                
            except Exception as e:
                logger.warning(f"Error processing {symbol}: {e}")
        
        return signals
    
    def _create_signal(self, symbol: str, apex_result, bars: List[OhlcvBar]) -> Dict[str, Any]:
        """Create a feedback signal record."""
        entry_price = bars[-1].close
        
        protocol_flags = []
        for p in apex_result.protocol_results:
            if hasattr(p, 'protocol_id') and hasattr(p, 'fired') and p.fired:
                protocol_flags.append(p.protocol_id)
        
        omega_flags = []
        if isinstance(apex_result.omega_overrides, dict):
            omega_flags = [k for k, v in apex_result.omega_overrides.items() if v]
        
        if apex_result.quantrascore >= 75:
            quality_tier = "A+"
        elif apex_result.quantrascore >= 65:
            quality_tier = "A"
        elif apex_result.quantrascore >= 50:
            quality_tier = "B"
        elif apex_result.quantrascore >= 35:
            quality_tier = "C"
        else:
            quality_tier = "D"
        
        return {
            "sample_id": f"alpha_factory_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "alpha_factory_demo",
            "quantra_score": apex_result.quantrascore,
            "risk_tier": apex_result.risk_tier.value,
            "regime": apex_result.regime.value,
            "entropy_state": apex_result.entropy_state.value,
            "suppression_state": apex_result.suppression_state.value,
            "drift_state": apex_result.drift_state.value,
            "protocol_flags": protocol_flags,
            "omega_flags": omega_flags,
            "entry_price": entry_price,
            "quality_tier": quality_tier,
            "signal_type": self._determine_signal_type(apex_result),
            "confidence": min(100, apex_result.quantrascore + 10),
        }
    
    def _determine_signal_type(self, apex_result) -> str:
        """Determine signal type based on analysis."""
        if apex_result.risk_tier.value == "extreme":
            return "avoid"
        if apex_result.quantrascore >= 70 and apex_result.regime.value in ["trending_up", "breakout"]:
            return "strong_buy"
        if apex_result.quantrascore >= 55:
            return "buy"
        if apex_result.quantrascore <= 30:
            return "sell"
        return "hold"
    
    def save_signals(self, signals: List[Dict[str, Any]]) -> int:
        """Save signals to feedback samples file."""
        existing = []
        
        if self.output_path.exists():
            with open(self.output_path, "r") as f:
                existing = json.load(f)
        
        all_samples = existing + signals
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(all_samples, f, indent=2, default=str)
        
        logger.info(f"Saved {len(all_samples)} total samples to {self.output_path}")
        return len(all_samples)
    
    def run_multiple_cycles(self, num_cycles: int = 5) -> Dict[str, Any]:
        """Run multiple cycles of signal generation."""
        all_signals = []
        
        for i in range(num_cycles):
            logger.info(f"Running Alpha Factory cycle {i+1}/{num_cycles}...")
            signals = self.run_cycle()
            all_signals.extend(signals)
        
        total_saved = self.save_signals(all_signals)
        
        signal_types = {}
        for s in all_signals:
            st = s.get("signal_type", "unknown")
            signal_types[st] = signal_types.get(st, 0) + 1
        
        return {
            "cycles_run": num_cycles,
            "signals_generated": len(all_signals),
            "total_samples": total_saved,
            "signal_distribution": signal_types,
        }


def main():
    parser = argparse.ArgumentParser(description="Run Alpha Factory Demo Mode")
    parser.add_argument("--cycles", type=int, default=3, help="Number of cycles to run")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    args = parser.parse_args()
    
    print("=" * 70)
    print("QuantraCore Apex - Alpha Factory Demo Mode")
    print("=" * 70)
    
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    
    factory = AlphaFactoryDemo(symbols=symbols)
    
    print(f"\nRunning {args.cycles} cycles for {len(factory.symbols)} symbols...")
    
    result = factory.run_multiple_cycles(args.cycles)
    
    print(f"\n{'=' * 70}")
    print("Alpha Factory Demo Results")
    print(f"{'=' * 70}")
    print(f"  Cycles run: {result['cycles_run']}")
    print(f"  Signals generated: {result['signals_generated']}")
    print(f"  Total feedback samples: {result['total_samples']}")
    print(f"\nSignal Distribution:")
    for signal_type, count in result['signal_distribution'].items():
        print(f"    {signal_type}: {count}")
    
    print(f"\n{'=' * 70}")
    print("Alpha Factory demo complete! Signals added to feedback loop.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
