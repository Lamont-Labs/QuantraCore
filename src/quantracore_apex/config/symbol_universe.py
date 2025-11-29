"""
Symbol Universe Loader for QuantraCore Apex.

Provides centralized access to the symbol universe configuration,
supporting all cap buckets from mega to penny stocks.

Version: 9.0-A
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import yaml


UNIVERSE_CONFIG_PATH = Path("config/symbol_universe.yaml")

SMALLCAP_BUCKETS = {"small", "micro", "nano", "penny"}
ALL_BUCKETS = {"mega", "large", "mid", "small", "micro", "nano", "penny"}


@dataclass
class SymbolInfo:
    """Information about a single symbol."""
    symbol: str
    name: str = ""
    asset_class: str = "equity"
    sector: str = ""
    market_cap_bucket: str = "unknown"
    float_millions: float = 0.0
    active: bool = True
    allow_smallcap_scan: bool = True
    notes: str = ""
    
    @property
    def is_smallcap(self) -> bool:
        """Check if symbol is in a small-cap bucket."""
        return self.market_cap_bucket in SMALLCAP_BUCKETS
    
    @property
    def risk_category(self) -> str:
        """Get risk category based on market cap bucket."""
        risk_map = {
            "mega": "low",
            "large": "low",
            "mid": "medium",
            "small": "high",
            "micro": "high",
            "nano": "extreme",
            "penny": "extreme",
        }
        return risk_map.get(self.market_cap_bucket, "high")


@dataclass
class UniverseConfig:
    """Loaded symbol universe configuration."""
    version: str = "9.0-A"
    symbols: Dict[str, SymbolInfo] = field(default_factory=dict)
    universes: Dict[str, List[str]] = field(default_factory=dict)
    bucket_definitions: Dict[str, dict] = field(default_factory=dict)
    

_universe_cache: Optional[UniverseConfig] = None


def _load_config() -> dict:
    """Load raw YAML configuration."""
    config_path = UNIVERSE_CONFIG_PATH
    if not config_path.exists():
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "symbol_universe.yaml"
    
    if not config_path.exists():
        return {"version": "9.0-A", "universes": {}}
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def load_symbol_universe(force_reload: bool = False) -> UniverseConfig:
    """
    Load the symbol universe configuration.
    
    Args:
        force_reload: Force reload from disk even if cached
        
    Returns:
        UniverseConfig with all symbols and universes
    """
    global _universe_cache
    
    if _universe_cache is not None and not force_reload:
        return _universe_cache
    
    raw_config = _load_config()
    
    config = UniverseConfig(
        version=raw_config.get("version", "9.0-A"),
        bucket_definitions=raw_config.get("market_cap_buckets", {}),
    )
    
    universes_raw = raw_config.get("universes", {})
    
    for universe_name, universe_data in universes_raw.items():
        if not isinstance(universe_data, dict):
            continue
            
        symbols_list = universe_data.get("symbols", [])
        universe_symbols = []
        
        for sym_data in symbols_list:
            if not isinstance(sym_data, dict):
                continue
                
            symbol = sym_data.get("symbol", "")
            if not symbol:
                continue
            
            info = SymbolInfo(
                symbol=symbol,
                name=sym_data.get("name", ""),
                asset_class=sym_data.get("asset_class", "equity"),
                sector=sym_data.get("sector", ""),
                market_cap_bucket=sym_data.get("market_cap_bucket", "unknown"),
                float_millions=float(sym_data.get("float_millions", 0)),
                active=sym_data.get("active", True),
                allow_smallcap_scan=sym_data.get("allow_smallcap_scan", True),
                notes=sym_data.get("notes", ""),
            )
            
            config.symbols[symbol] = info
            if info.active:
                universe_symbols.append(symbol)
        
        config.universes[universe_name] = universe_symbols
    
    _universe_cache = config
    return config


def get_all_symbols(active_only: bool = True) -> List[str]:
    """
    Get all symbols from all universes.
    
    Args:
        active_only: Only return active symbols
        
    Returns:
        List of symbol strings
    """
    config = load_symbol_universe()
    
    if active_only:
        return [s for s, info in config.symbols.items() if info.active]
    return list(config.symbols.keys())


def get_symbols_by_bucket(
    buckets: List[str],
    active_only: bool = True
) -> List[str]:
    """
    Get symbols filtered by market cap bucket.
    
    Args:
        buckets: List of bucket names (mega, large, mid, small, micro, nano, penny)
        active_only: Only return active symbols
        
    Returns:
        List of symbol strings matching the buckets
    """
    config = load_symbol_universe()
    bucket_set = set(buckets)
    
    result = []
    for symbol, info in config.symbols.items():
        if info.market_cap_bucket in bucket_set:
            if not active_only or info.active:
                result.append(symbol)
    
    return result


def get_symbols_for_mode(mode: str) -> List[str]:
    """
    Get symbols appropriate for a scan mode.
    
    Args:
        mode: Scan mode name (e.g., "full_us_equities", "high_vol_small_caps")
        
    Returns:
        List of symbol strings for the mode
    """
    config = load_symbol_universe()
    
    mode_to_buckets = {
        "full_us_equities": list(ALL_BUCKETS),
        "high_vol_small_caps": ["small", "micro", "nano", "penny"],
        "low_float_runners": ["micro", "nano", "penny"],
        "mega_large_focus": ["mega", "large"],
        "mid_cap_focus": ["mid"],
        "momentum_runners": ["mega", "large", "mid", "small", "micro"],
        "demo": ["mega", "large"],
        "ci_test": ["mega"],
    }
    
    mode_to_universe = {
        "demo": "demo",
        "ci_test": "demo",
    }
    
    if mode in mode_to_universe:
        universe_name = mode_to_universe[mode]
        return config.universes.get(universe_name, [])
    
    buckets = mode_to_buckets.get(mode, list(ALL_BUCKETS))
    return get_symbols_by_bucket(buckets)


def get_symbol_info(symbol: str) -> Optional[SymbolInfo]:
    """
    Get detailed information about a symbol.
    
    Args:
        symbol: Ticker symbol
        
    Returns:
        SymbolInfo or None if not found
    """
    config = load_symbol_universe()
    return config.symbols.get(symbol)


def is_smallcap(symbol: str) -> bool:
    """
    Check if a symbol is in a small-cap bucket.
    
    Args:
        symbol: Ticker symbol
        
    Returns:
        True if symbol is small/micro/nano/penny cap
    """
    info = get_symbol_info(symbol)
    if info is None:
        return False
    return info.is_smallcap


def get_universe_symbols(universe_name: str) -> List[str]:
    """
    Get all symbols from a named universe.
    
    Args:
        universe_name: Name of the universe (e.g., "demo", "smallcap_demo")
        
    Returns:
        List of symbol strings
    """
    config = load_symbol_universe()
    return config.universes.get(universe_name, [])


def get_smallcap_symbols(active_only: bool = True) -> List[str]:
    """
    Get all symbols in small-cap buckets.
    
    Args:
        active_only: Only return active symbols
        
    Returns:
        List of small/micro/nano/penny cap symbols
    """
    return get_symbols_by_bucket(list(SMALLCAP_BUCKETS), active_only=active_only)


def get_low_float_symbols(max_float_millions: float = 15.0) -> List[str]:
    """
    Get symbols with low float.
    
    Args:
        max_float_millions: Maximum float in millions
        
    Returns:
        List of low-float symbols
    """
    config = load_symbol_universe()
    
    return [
        symbol for symbol, info in config.symbols.items()
        if info.active and 0 < info.float_millions <= max_float_millions
    ]
