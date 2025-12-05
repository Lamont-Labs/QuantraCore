"""
Universe Scanner - Expands stock scanning from 26 to 2,000+ symbols.

Uses Alpaca's multi-symbol bars endpoint to efficiently prefilter
the entire market and find stocks with momentum/volume surges.

This replaces the hardcoded 26-stock QUICK_SCAN_UNIVERSE with a
dynamic scanning approach that covers thousands of stocks.
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

UNIVERSE_CACHE_FILE = Path("data/universes/liquid_universe.json")
UNIVERSE_CACHE_TTL_HOURS = 24

FULL_SYMBOL_LIST = Path("data/all_symbols.txt")

MIN_PRICE = 0.50
MAX_PRICE = 50.0
MIN_DOLLAR_VOLUME = 500_000
MIN_CHANGE_PCT = 2.0
MIN_VOLUME_SURGE = 1.5


class UniverseScanner:
    """
    Scans thousands of stocks to find hot candidates.
    
    Two-stage approach:
    1. Build liquid universe (daily): Filter 10,000+ symbols to ~2,000 tradeable stocks
    2. Intraday prefilter: Use Alpaca bulk bars to find ~100-300 hot stocks
    3. Run ML models on hot stocks only
    """
    
    def __init__(self):
        self._alpaca = None
        self._liquid_universe: List[str] = []
        self._last_universe_load = None
        self._hotlist: List[Dict] = []
        self._last_prefilter = None
        
    @property
    def alpaca(self):
        if self._alpaca is None:
            from src.quantracore_apex.data_layer.adapters.alpaca_data_adapter import AlpacaDataAdapter
            self._alpaca = AlpacaDataAdapter()
        return self._alpaca
    
    def load_all_symbols(self) -> List[str]:
        """Load all available symbols from master list."""
        if not FULL_SYMBOL_LIST.exists():
            logger.warning(f"Symbol list not found: {FULL_SYMBOL_LIST}")
            return self._get_fallback_universe()
        
        try:
            with open(FULL_SYMBOL_LIST, 'r') as f:
                symbols = [line.strip().upper() for line in f if line.strip()]
            
            valid_symbols = [
                s for s in symbols 
                if s.isalpha() and 1 <= len(s) <= 5
            ]
            
            logger.info(f"[UniverseScanner] Loaded {len(valid_symbols)} symbols from master list")
            return valid_symbols
        except Exception as e:
            logger.error(f"[UniverseScanner] Failed to load symbols: {e}")
            return self._get_fallback_universe()
    
    def _get_fallback_universe(self) -> List[str]:
        """Fallback universe if master list unavailable."""
        return [
            "QUBT", "RGTI", "QBTS", "IONQ", "BBAI", "SOUN",
            "MARA", "RIOT", "BITF", "CLSK", "HIVE",
            "BYND", "LCID", "HIMS", "NVTS", "SYM",
            "PLUG", "FCEL", "QS", "BLNK", "CHPT",
            "SOFI", "FUBO", "OPEN", "SMCI", "COIN",
            "AMC", "GME", "PLTR", "NIO", "XPEV",
            "RIVN", "LAZR", "CLOV", "GRAB", "DKNG",
            "HOOD", "UPST", "AFRM", "RBLX", "SNAP",
            "PINS", "ETSY", "ROKU", "CRWD", "ZS",
            "NET", "DDOG", "MDB", "SNOW", "PLBY"
        ]
    
    def build_liquid_universe(self, force_refresh: bool = False) -> List[str]:
        """
        Build universe of liquid, tradeable stocks.
        
        Filters:
        - Price: $0.50 - $50 (sweet spot for runners)
        - Average dollar volume > $500k/day
        - Valid US equities (no ETFs, ADRs with weird suffixes)
        
        Returns ~1,500-2,000 symbols.
        """
        if not force_refresh and UNIVERSE_CACHE_FILE.exists():
            try:
                cache_time = datetime.fromtimestamp(UNIVERSE_CACHE_FILE.stat().st_mtime)
                if datetime.now() - cache_time < timedelta(hours=UNIVERSE_CACHE_TTL_HOURS):
                    with open(UNIVERSE_CACHE_FILE, 'r') as f:
                        cached = json.load(f)
                    self._liquid_universe = cached.get('symbols', [])
                    logger.info(f"[UniverseScanner] Loaded {len(self._liquid_universe)} symbols from cache")
                    return self._liquid_universe
            except Exception as e:
                logger.warning(f"[UniverseScanner] Cache read failed: {e}")
        
        all_symbols = self.load_all_symbols()
        logger.info(f"[UniverseScanner] Building liquid universe from {len(all_symbols)} symbols...")
        
        excluded_suffixes = ['W', 'U', 'R', 'P', 'Q', 'L']
        filtered_symbols = [
            s for s in all_symbols
            if not any(s.endswith(suf) for suf in excluded_suffixes)
            and 2 <= len(s) <= 5
        ]
        
        logger.info(f"[UniverseScanner] After suffix filter: {len(filtered_symbols)} symbols")
        
        end = datetime.now()
        start = end - timedelta(days=5)
        
        liquid_symbols = []
        batch_size = 100
        
        for i in range(0, len(filtered_symbols), batch_size):
            batch = filtered_symbols[i:i + batch_size]
            
            try:
                bars_data = self.alpaca.get_multi_bars(batch, start, end, "1d")
                
                for symbol, bars in bars_data.items():
                    if len(bars) < 2:
                        continue
                    
                    latest = bars[-1]
                    price = latest.close
                    volume = latest.volume
                    dollar_volume = price * volume
                    
                    if MIN_PRICE <= price <= MAX_PRICE and dollar_volume >= MIN_DOLLAR_VOLUME:
                        liquid_symbols.append({
                            'symbol': symbol,
                            'price': price,
                            'volume': volume,
                            'dollar_volume': dollar_volume
                        })
                
                if i % 500 == 0 and i > 0:
                    logger.info(f"[UniverseScanner] Processed {i}/{len(filtered_symbols)}, found {len(liquid_symbols)} liquid")
                    
            except Exception as e:
                logger.warning(f"[UniverseScanner] Batch {i} failed: {e}")
                continue
        
        liquid_symbols.sort(key=lambda x: x['dollar_volume'], reverse=True)
        
        top_liquid = liquid_symbols[:2000]
        self._liquid_universe = [s['symbol'] for s in top_liquid]
        
        UNIVERSE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(UNIVERSE_CACHE_FILE, 'w') as f:
            json.dump({
                'symbols': self._liquid_universe,
                'built_at': datetime.now().isoformat(),
                'total_scanned': len(filtered_symbols),
                'liquid_found': len(liquid_symbols)
            }, f, indent=2)
        
        logger.info(f"[UniverseScanner] Built liquid universe: {len(self._liquid_universe)} symbols")
        return self._liquid_universe
    
    def get_liquid_universe(self) -> List[str]:
        """Get cached liquid universe or build if needed."""
        if self._liquid_universe:
            return self._liquid_universe
        return self.build_liquid_universe()
    
    def prefilter_for_momentum(
        self,
        symbols: Optional[List[str]] = None,
        min_change_pct: float = MIN_CHANGE_PCT,
        min_volume_surge: float = MIN_VOLUME_SURGE,
        max_results: int = 300
    ) -> List[Dict[str, Any]]:
        """
        Fast prefilter to find stocks with momentum/volume surges.
        
        Uses Alpaca multi-symbol bars to efficiently scan 2,000 stocks
        in ~10 API calls (200 symbols per call).
        
        Returns:
            List of hot stocks sorted by momentum score
        """
        if symbols is None:
            symbols = self.get_liquid_universe()
        
        if not symbols:
            logger.warning("[UniverseScanner] No symbols to prefilter")
            return []
        
        logger.info(f"[UniverseScanner] Prefiltering {len(symbols)} symbols for momentum...")
        
        end = datetime.now()
        start = end - timedelta(days=5)
        
        hot_stocks = []
        batch_size = 200
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            try:
                bars_data = self.alpaca.get_multi_bars(batch, start, end, "1d")
                
                for symbol, bars in bars_data.items():
                    if len(bars) < 3:
                        continue
                    
                    try:
                        latest = bars[-1]
                        prev = bars[-2]
                        
                        prices = [b.close for b in bars]
                        volumes = [b.volume for b in bars]
                        
                        current_price = latest.close
                        prev_close = prev.close
                        
                        if prev_close == 0:
                            continue
                        
                        day_change_pct = ((current_price - prev_close) / prev_close) * 100
                        
                        avg_volume = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else volumes[0]
                        volume_surge = latest.volume / avg_volume if avg_volume > 0 else 1.0
                        
                        week_low = min(b.low for b in bars)
                        week_high = max(b.high for b in bars)
                        week_range = week_high - week_low
                        position_in_range = (current_price - week_low) / week_range if week_range > 0 else 0.5
                        
                        momentum_score = (
                            day_change_pct * 0.4 +
                            (volume_surge - 1) * 10 * 0.3 +
                            position_in_range * 10 * 0.3
                        )
                        
                        if day_change_pct >= min_change_pct or volume_surge >= min_volume_surge:
                            hot_stocks.append({
                                'symbol': symbol,
                                'price': current_price,
                                'day_change_pct': round(day_change_pct, 2),
                                'volume_surge': round(volume_surge, 2),
                                'position_in_range': round(position_in_range, 2),
                                'momentum_score': round(momentum_score, 2),
                                'volume': latest.volume,
                                'dollar_volume': current_price * latest.volume
                            })
                    except Exception as e:
                        continue
                        
            except Exception as e:
                logger.warning(f"[UniverseScanner] Prefilter batch {i} failed: {e}")
                continue
        
        hot_stocks.sort(key=lambda x: x['momentum_score'], reverse=True)
        
        result = hot_stocks[:max_results]
        self._hotlist = result
        self._last_prefilter = datetime.now()
        
        logger.info(f"[UniverseScanner] Prefilter found {len(result)} hot stocks from {len(symbols)} scanned")
        
        return result
    
    def get_hotlist_symbols(self) -> List[str]:
        """Get just the symbols from the current hotlist."""
        return [s['symbol'] for s in self._hotlist]
    
    def get_scan_universe(
        self,
        quick_scan: bool = False,
        prefilter: bool = True,
        min_change_pct: float = 1.5,
        min_volume_surge: float = 1.3,
        max_results: int = 300
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Get the universe of symbols to scan with ML models.
        
        Args:
            quick_scan: If True, use small fallback universe (26 stocks)
            prefilter: If True, run momentum prefilter on liquid universe
            min_change_pct: Minimum % change for prefilter
            min_volume_surge: Minimum volume surge ratio
            max_results: Maximum stocks to pass to ML models
            
        Returns:
            Tuple of (symbols list, scan metadata)
        """
        if quick_scan:
            symbols = self._get_fallback_universe()
            return symbols, {
                'mode': 'quick_scan',
                'symbols_count': len(symbols),
                'prefiltered': False
            }
        
        liquid = self.get_liquid_universe()
        
        if not liquid:
            logger.warning("[UniverseScanner] No liquid universe - falling back to quick scan")
            symbols = self._get_fallback_universe()
            return symbols, {
                'mode': 'fallback',
                'symbols_count': len(symbols),
                'prefiltered': False
            }
        
        if prefilter:
            hot_stocks = self.prefilter_for_momentum(
                symbols=liquid,
                min_change_pct=min_change_pct,
                min_volume_surge=min_volume_surge,
                max_results=max_results
            )
            
            if hot_stocks:
                symbols = [s['symbol'] for s in hot_stocks]
                return symbols, {
                    'mode': 'prefiltered',
                    'liquid_universe_size': len(liquid),
                    'symbols_count': len(symbols),
                    'prefiltered': True,
                    'top_movers': hot_stocks[:10]
                }
        
        symbols = liquid[:max_results]
        return symbols, {
            'mode': 'liquid_top',
            'liquid_universe_size': len(liquid),
            'symbols_count': len(symbols),
            'prefiltered': False
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get scanner status."""
        return {
            'liquid_universe_size': len(self._liquid_universe),
            'hotlist_size': len(self._hotlist),
            'last_prefilter': self._last_prefilter.isoformat() if self._last_prefilter else None,
            'cache_file': str(UNIVERSE_CACHE_FILE),
            'cache_exists': UNIVERSE_CACHE_FILE.exists(),
            'alpaca_available': self.alpaca.is_available() if self._alpaca else 'not_initialized'
        }


_scanner_instance: Optional[UniverseScanner] = None


def get_universe_scanner() -> UniverseScanner:
    """Get singleton universe scanner instance."""
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = UniverseScanner()
    return _scanner_instance
