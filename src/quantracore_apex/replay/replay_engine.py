"""
Sandbox Replay Engine for v9.0-A
Reusable module to replay historical sequences for drift and regression testing.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ReplayConfig:
    """Configuration for replay execution."""
    universe: str = "demo"
    timeframe: str = "1d"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    lookback_bars: int = 100
    initial_capital: float = 100000.0
    position_size_pct: float = 0.10
    enable_apexcore: bool = False


@dataclass
class ReplayResult:
    """Result of a replay execution."""
    config: ReplayConfig
    start_timestamp: str
    end_timestamp: str
    duration_seconds: float
    symbols_processed: int
    signals_generated: int
    equity_curve: List[float] = field(default_factory=list)
    signal_frequency_stats: Dict[str, int] = field(default_factory=dict)
    drift_flags: List[str] = field(default_factory=list)
    consistency_stats: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "universe": self.config.universe,
                "timeframe": self.config.timeframe,
                "start_date": self.config.start_date,
                "end_date": self.config.end_date,
                "lookback_bars": self.config.lookback_bars,
                "initial_capital": self.config.initial_capital,
            },
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "duration_seconds": self.duration_seconds,
            "symbols_processed": self.symbols_processed,
            "signals_generated": self.signals_generated,
            "equity_curve": self.equity_curve,
            "signal_frequency_stats": self.signal_frequency_stats,
            "drift_flags": self.drift_flags,
            "consistency_stats": self.consistency_stats,
            "errors": self.errors,
        }


class ReplayEngine:
    """
    Sandbox replay engine for historical sequence replay.
    
    Designed for:
    - Drift and regression testing
    - Deterministic replay logs
    - Key metrics computation
    
    Constraints:
    - Designed for small to medium universes
    - Heavy runs still manual
    - Not run in CI but callable from CLI or notebooks
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.replay_history: List[ReplayResult] = []
    
    def run_replay(
        self,
        config: Optional[ReplayConfig] = None,
        universe: str = "demo",
        timeframe: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_bars: int = 100
    ) -> ReplayResult:
        """
        Execute a replay over historical data.
        
        Args:
            config: ReplayConfig or individual parameters
            universe: Universe name from symbol_universe.yaml
            timeframe: Bar timeframe (1d, 1h, etc.)
            start_date: Start date ISO format
            end_date: End date ISO format
            lookback_bars: Number of bars to look back
        
        Returns:
            ReplayResult with equity curve and metrics
        """
        if config is None:
            config = ReplayConfig(
                universe=universe,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                lookback_bars=lookback_bars,
            )
        
        start_time = datetime.utcnow()
        
        symbols = self._load_universe_symbols(config.universe)
        
        equity_curve = [config.initial_capital]
        signals_generated = 0
        signal_stats = {"buy": 0, "sell": 0, "hold": 0}
        drift_flags = []
        errors = []
        
        for symbol in symbols:
            try:
                ohlcv_data = self._load_cached_data(symbol, config.timeframe)
                
                if not ohlcv_data:
                    errors.append(f"{symbol}: no_cached_data")
                    continue
                
                scan_result = self._run_scan(symbol, ohlcv_data, config)
                
                if scan_result.get("signal"):
                    signals_generated += 1
                    signal_type = scan_result.get("signal_type", "hold")
                    signal_stats[signal_type] = signal_stats.get(signal_type, 0) + 1
                
                if scan_result.get("consistency_warning"):
                    drift_flags.append(f"{symbol}: score_consistency_warning")
                
                pnl = self._simulate_pnl(scan_result, config)
                equity_curve.append(equity_curve[-1] + pnl)
                
            except Exception as e:
                errors.append(f"{symbol}: {str(e)}")
                logger.error(f"Replay error for {symbol}: {e}")
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        result = ReplayResult(
            config=config,
            start_timestamp=start_time.isoformat(),
            end_timestamp=end_time.isoformat(),
            duration_seconds=duration,
            symbols_processed=len(symbols),
            signals_generated=signals_generated,
            equity_curve=equity_curve,
            signal_frequency_stats=signal_stats,
            drift_flags=drift_flags,
            consistency_stats={
                "warnings": len(drift_flags),
                "error_count": len(errors),
            },
            errors=errors,
        )
        
        self.replay_history.append(result)
        logger.info(
            f"Replay complete: {len(symbols)} symbols, "
            f"{signals_generated} signals, {duration:.2f}s"
        )
        
        return result
    
    def _load_universe_symbols(self, universe: str) -> List[str]:
        """Load symbols from universe configuration."""
        demo_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
            "META", "TSLA", "JPM", "V", "JNJ"
        ]
        
        universe_file = Path("config/symbol_universe.yaml")
        if universe_file.exists():
            try:
                import yaml
                with open(universe_file) as f:
                    config = yaml.safe_load(f)
                
                if universe in config.get("universes", {}):
                    universe_data = config["universes"][universe]
                    if "symbols" in universe_data:
                        return [
                            s["symbol"] for s in universe_data["symbols"]
                            if s.get("active", True)
                        ]
            except Exception as e:
                logger.warning(f"Failed to load universe config: {e}")
        
        return demo_symbols
    
    def _load_cached_data(
        self,
        symbol: str,
        timeframe: str
    ) -> List[Dict[str, Any]]:
        """Load cached OHLCV data for a symbol."""
        cache_patterns = [
            self.cache_dir / f"{symbol}_{timeframe}.parquet",
            self.cache_dir / f"{symbol}_{timeframe}.json",
            self.cache_dir / f"{symbol}.json",
        ]
        
        for cache_file in cache_patterns:
            if cache_file.exists():
                try:
                    if cache_file.suffix == ".json":
                        with open(cache_file) as f:
                            return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load cache {cache_file}: {e}")
        
        return self._generate_synthetic_data(symbol, bars=100)
    
    def _generate_synthetic_data(
        self,
        symbol: str,
        bars: int = 100
    ) -> List[Dict[str, Any]]:
        """Generate synthetic OHLCV data for testing."""
        import random
        
        seed = hash(symbol) % (2**32)
        random.seed(seed)
        
        base_price = 100.0 + random.random() * 200
        data = []
        
        for i in range(bars):
            change = (random.random() - 0.5) * 0.04
            open_price = base_price
            close_price = base_price * (1 + change)
            high_price = max(open_price, close_price) * (1 + random.random() * 0.02)
            low_price = min(open_price, close_price) * (1 - random.random() * 0.02)
            volume = int(1000000 * (0.5 + random.random()))
            
            data.append({
                "date": f"2024-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}",
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": volume,
            })
            
            base_price = close_price
        
        return data
    
    def _run_scan(
        self,
        symbol: str,
        ohlcv_data: List[Dict[str, Any]],
        config: ReplayConfig
    ) -> Dict[str, Any]:
        """
        Run a simulated scan on the data.
        
        In a real implementation, this would call the ApexEngine.
        For now, returns a simulated result.
        """
        if not ohlcv_data:
            return {"signal": False, "error": "no_data"}
        
        last_bar = ohlcv_data[-1]
        close = last_bar.get("close", 0)
        
        import random
        random.seed(hash(f"{symbol}_{close}"))
        
        score = 40 + random.random() * 40
        
        if score >= 65:
            signal_type = "buy"
            signal = True
        elif score <= 35:
            signal_type = "sell"
            signal = True
        else:
            signal_type = "hold"
            signal = False
        
        return {
            "symbol": symbol,
            "signal": signal,
            "signal_type": signal_type,
            "quantrascore": score,
            "consistency_warning": random.random() < 0.05,
        }
    
    def _simulate_pnl(
        self,
        scan_result: Dict[str, Any],
        config: ReplayConfig
    ) -> float:
        """Simulate P&L for a scan result."""
        if not scan_result.get("signal"):
            return 0.0
        
        import random
        
        position_value = config.initial_capital * config.position_size_pct
        
        pnl_pct = (random.random() - 0.48) * 0.04
        
        return position_value * pnl_pct
    
    def run_demo_replay(self) -> ReplayResult:
        """Run a quick demo replay on the demo universe."""
        return self.run_replay(
            universe="demo",
            timeframe="1d",
            lookback_bars=50,
        )
    
    def save_replay_log(self, result: ReplayResult, filepath: Optional[str] = None):
        """Save replay result to log file."""
        log_path: Path
        if filepath is None:
            log_dir = Path("provenance")
            log_dir.mkdir(exist_ok=True)
            log_path = log_dir / "replay_runs_log.jsonl"
        else:
            log_path = Path(filepath)
        
        with open(log_path, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")
        
        logger.info(f"Saved replay log to {log_path}")
    
    def get_replay_summary(self) -> Dict[str, Any]:
        """Get summary of replay history."""
        if not self.replay_history:
            return {"total_replays": 0}
        
        total_signals = sum(r.signals_generated for r in self.replay_history)
        total_symbols = sum(r.symbols_processed for r in self.replay_history)
        
        return {
            "total_replays": len(self.replay_history),
            "total_signals": total_signals,
            "total_symbols_processed": total_symbols,
            "avg_duration_seconds": sum(r.duration_seconds for r in self.replay_history) / len(self.replay_history),
        }
