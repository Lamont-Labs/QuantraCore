"""
Universe Scanner for QuantraCore Apex.

Provides batched scanning of large symbol universes with
K6-safe performance optimizations.

Version: 9.0-A
"""

import time
import logging
from typing import List, Dict, Optional, Any, Generator, Callable
from dataclasses import dataclass, field
from datetime import datetime

from src.quantracore_apex.config.symbol_universe import (
    get_symbols_for_mode,
    get_symbol_info,
)
from src.quantracore_apex.config.scan_modes import (
    load_scan_mode,
    ScanModeConfig,
    get_performance_config,
)

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Result of scanning a single symbol."""
    symbol: str
    quantrascore: float = 0.0
    score_bucket: str = "unknown"
    regime: str = "unknown"
    risk_tier: str = "unknown"
    entropy_state: str = "unknown"
    suppression_state: str = "unknown"
    drift_state: str = "unknown"
    verdict_action: str = "unknown"
    verdict_confidence: float = 0.0
    omega_alerts: List[str] = field(default_factory=list)
    protocol_fired_count: int = 0
    window_hash: str = ""
    timestamp: str = ""
    
    market_cap_bucket: str = "unknown"
    smallcap_flag: bool = False
    mr_fuse_score: float = 0.0
    speculative_flag: bool = False
    float_millions: float = 0.0
    
    error: Optional[str] = None
    scan_time_ms: float = 0.0


@dataclass
class UniverseScanResult:
    """Result of scanning an entire universe."""
    mode: str
    scan_count: int = 0
    success_count: int = 0
    error_count: int = 0
    results: List[ScanResult] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)
    timestamp: str = ""
    total_time_seconds: float = 0.0
    chunks_processed: int = 0
    
    smallcap_count: int = 0
    extreme_risk_count: int = 0
    runner_candidate_count: int = 0


class UniverseScanner:
    """
    Batched universe scanner with K6-safe performance.
    
    Features:
    - Chunked processing to limit memory usage
    - Configurable batch delays for CPU breathing
    - Progress callbacks for UI updates
    - Automatic caching integration
    - Small-cap and MonsterRunner awareness
    """
    
    def __init__(
        self,
        data_client: Optional[Any] = None,
        engine: Optional[Any] = None,
        max_concurrent: int = 4,
    ):
        self.data_client = data_client
        self.engine = engine
        self.max_concurrent = max_concurrent
        self.perf_config = get_performance_config()
    
    def _get_symbols_for_mode(
        self,
        mode_config: ScanModeConfig
    ) -> List[str]:
        """Get symbols for the scan mode."""
        if mode_config.use_universe:
            from src.quantracore_apex.config.symbol_universe import get_universe_symbols
            symbols = get_universe_symbols(mode_config.use_universe)
        else:
            symbols = get_symbols_for_mode(mode_config.name)
        
        if mode_config.max_symbols and len(symbols) > mode_config.max_symbols:
            symbols = symbols[:mode_config.max_symbols]
        
        return symbols
    
    def _chunk_symbols(
        self,
        symbols: List[str],
        chunk_size: int
    ) -> Generator[List[str], None, None]:
        """Split symbols into chunks."""
        for i in range(0, len(symbols), chunk_size):
            yield symbols[i:i + chunk_size]
    
    def _scan_symbol(
        self,
        symbol: str,
        lookback_days: int = 250,
        timeframe: str = "1d",
    ) -> ScanResult:
        """Scan a single symbol."""
        start_time = time.time()
        
        try:
            symbol_info = get_symbol_info(symbol)
            
            if self.data_client is None:
                from src.quantracore_apex.data_layer.client import create_data_client
                self.data_client = create_data_client()
            
            fetch_result = self.data_client.fetch(
                symbol=symbol,
                days=lookback_days,
                timeframe="day",
            )
            
            if not fetch_result.bars or len(fetch_result.bars) < 50:
                return ScanResult(
                    symbol=symbol,
                    error=f"Insufficient data: {len(fetch_result.bars) if fetch_result.bars else 0} bars",
                    scan_time_ms=(time.time() - start_time) * 1000,
                )
            
            if self.engine is None:
                from src.quantracore_apex.core.engine import ApexEngine
                self.engine = ApexEngine()
            
            engine_result = self.engine.run_scan(
                bars=fetch_result.bars,
                symbol=symbol,
                timeframe=timeframe
            )
            
            from src.quantracore_apex.protocols.monster_runner.fuse_score import (
                calculate_mr_fuse_score
            )
            
            mr_fuse = calculate_mr_fuse_score(
                bars=fetch_result.bars,
                symbol_info=symbol_info,
                engine_result=engine_result,
            )
            
            is_small = symbol_info.is_smallcap if symbol_info else False
            market_cap_bucket = symbol_info.market_cap_bucket if symbol_info else "unknown"
            float_m = symbol_info.float_millions if symbol_info else 0.0
            
            speculative = (
                is_small or
                market_cap_bucket in ["nano", "penny"] or
                engine_result.risk_tier == "extreme"
            )
            
            scan_time = (time.time() - start_time) * 1000
            
            omega_alerts = [k for k, v in engine_result.omega_overrides.items() if v]
            
            protocol_fired_count = len([p for p in engine_result.protocol_results if p.fired])
            
            return ScanResult(
                symbol=symbol,
                quantrascore=engine_result.quantrascore,
                score_bucket=engine_result.score_bucket,
                regime=engine_result.regime,
                risk_tier=engine_result.risk_tier,
                entropy_state=engine_result.entropy_state,
                suppression_state=engine_result.suppression_state,
                drift_state=engine_result.drift_state,
                verdict_action=engine_result.verdict.action,
                verdict_confidence=engine_result.verdict.confidence,
                omega_alerts=omega_alerts,
                protocol_fired_count=protocol_fired_count,
                window_hash=engine_result.window_hash,
                timestamp=datetime.now().isoformat(),
                market_cap_bucket=market_cap_bucket,
                smallcap_flag=is_small,
                mr_fuse_score=mr_fuse.fuse_score,
                speculative_flag=speculative,
                float_millions=float_m,
                scan_time_ms=scan_time,
            )
            
        except Exception as e:
            return ScanResult(
                symbol=symbol,
                error=str(e),
                scan_time_ms=(time.time() - start_time) * 1000,
            )
    
    def scan(
        self,
        mode: str = "mega_large_focus",
        lookback_days: int = 250,
        timeframe: str = "1d",
        max_symbols: Optional[int] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
        on_chunk_complete: Optional[Callable[[int, int], None]] = None,
    ) -> UniverseScanResult:
        """
        Scan a universe of symbols.
        
        Args:
            mode: Scan mode name
            lookback_days: Days of historical data
            timeframe: Bar timeframe
            max_symbols: Override max symbols from mode config
            on_progress: Callback(symbol, current, total)
            on_chunk_complete: Callback(chunk_num, total_chunks)
            
        Returns:
            UniverseScanResult with all scan results
        """
        start_time = time.time()
        
        try:
            mode_config = load_scan_mode(mode)
        except ValueError:
            mode_config = ScanModeConfig(
                name=mode,
                buckets=["mega", "large"],
                max_symbols=max_symbols or 100,
                chunk_size=50,
            )
        
        symbols = self._get_symbols_for_mode(mode_config)
        
        if max_symbols:
            symbols = symbols[:max_symbols]
        
        result = UniverseScanResult(
            mode=mode,
            scan_count=len(symbols),
            timestamp=datetime.now().isoformat(),
        )
        
        if not symbols:
            return result
        
        chunk_size = mode_config.chunk_size
        chunks = list(self._chunk_symbols(symbols, chunk_size))
        total_chunks = len(chunks)
        
        processed = 0
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_results = []
            
            for symbol in chunk:
                scan_result = self._scan_symbol(
                    symbol=symbol,
                    lookback_days=lookback_days,
                    timeframe=timeframe,
                )
                chunk_results.append(scan_result)
                
                processed += 1
                if on_progress:
                    on_progress(symbol, processed, len(symbols))
            
            for sr in chunk_results:
                if sr.error:
                    result.error_count += 1
                    result.errors.append({
                        "symbol": sr.symbol,
                        "error": sr.error,
                    })
                else:
                    result.success_count += 1
                    result.results.append(sr)
                    
                    if sr.smallcap_flag:
                        result.smallcap_count += 1
                    if sr.risk_tier == "extreme":
                        result.extreme_risk_count += 1
                    if sr.mr_fuse_score >= 60:
                        result.runner_candidate_count += 1
            
            result.chunks_processed = chunk_idx + 1
            
            if on_chunk_complete:
                on_chunk_complete(chunk_idx + 1, total_chunks)
            
            if chunk_idx < total_chunks - 1 and mode_config.batch_delay_seconds > 0:
                time.sleep(mode_config.batch_delay_seconds)
        
        result.results.sort(key=lambda x: x.quantrascore, reverse=True)
        
        result.total_time_seconds = time.time() - start_time
        
        return result
    
    def scan_symbols(
        self,
        symbols: List[str],
        lookback_days: int = 250,
        timeframe: str = "1d",
        chunk_size: int = 50,
        batch_delay: float = 0.2,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> UniverseScanResult:
        """
        Scan a specific list of symbols (not using a mode).
        
        Args:
            symbols: List of ticker symbols
            lookback_days: Days of historical data
            timeframe: Bar timeframe
            chunk_size: Symbols per batch
            batch_delay: Delay between batches
            on_progress: Progress callback
            
        Returns:
            UniverseScanResult
        """
        start_time = time.time()
        
        result = UniverseScanResult(
            mode="custom",
            scan_count=len(symbols),
            timestamp=datetime.now().isoformat(),
        )
        
        chunks = list(self._chunk_symbols(symbols, chunk_size))
        processed = 0
        
        for chunk_idx, chunk in enumerate(chunks):
            for symbol in chunk:
                scan_result = self._scan_symbol(
                    symbol=symbol,
                    lookback_days=lookback_days,
                    timeframe=timeframe,
                )
                
                if scan_result.error:
                    result.error_count += 1
                    result.errors.append({
                        "symbol": scan_result.symbol,
                        "error": scan_result.error,
                    })
                else:
                    result.success_count += 1
                    result.results.append(scan_result)
                    
                    if scan_result.smallcap_flag:
                        result.smallcap_count += 1
                    if scan_result.risk_tier == "extreme":
                        result.extreme_risk_count += 1
                    if scan_result.mr_fuse_score >= 60:
                        result.runner_candidate_count += 1
                
                processed += 1
                if on_progress:
                    on_progress(symbol, processed, len(symbols))
            
            result.chunks_processed = chunk_idx + 1
            
            if chunk_idx < len(chunks) - 1 and batch_delay > 0:
                time.sleep(batch_delay)
        
        result.results.sort(key=lambda x: x.quantrascore, reverse=True)
        result.total_time_seconds = time.time() - start_time
        
        return result


def create_universe_scanner(
    data_client: Optional[Any] = None,
    engine: Optional[Any] = None,
) -> UniverseScanner:
    """
    Factory function to create a UniverseScanner.
    
    Args:
        data_client: Optional DataClient instance
        engine: Optional ApexEngine instance
        
    Returns:
        Configured UniverseScanner
    """
    return UniverseScanner(data_client=data_client, engine=engine)


def quick_scan(
    mode: str = "demo",
    max_symbols: int = 10,
    lookback_days: int = 250,
) -> UniverseScanResult:
    """
    Quick convenience function for simple scans.
    
    Args:
        mode: Scan mode name
        max_symbols: Maximum symbols to scan
        lookback_days: Days of history
        
    Returns:
        UniverseScanResult
    """
    scanner = create_universe_scanner()
    return scanner.scan(
        mode=mode,
        max_symbols=max_symbols,
        lookback_days=lookback_days,
    )
