"""
Multi-Asset Configuration for QuantraCore Apex.

Defines asset classes, their characteristics, and trading parameters.
Supports stocks, ETFs, and crypto - LONG ONLY.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class AssetType(Enum):
    """Supported asset types."""
    STOCK = "stock"
    ETF = "etf"
    CRYPTO = "crypto"


@dataclass
class AssetConfig:
    """Configuration for an asset type."""
    asset_type: AssetType
    max_position_pct: float
    default_stop_loss_pct: float
    default_take_profit_pct: float
    max_positions: int
    volatility_multiplier: float
    trading_hours: str
    min_price: float = 0.01
    max_price: float = 100000.0
    
    
ASSET_CONFIGS: Dict[AssetType, AssetConfig] = {
    AssetType.STOCK: AssetConfig(
        asset_type=AssetType.STOCK,
        max_position_pct=0.10,
        default_stop_loss_pct=0.08,
        default_take_profit_pct=0.50,
        max_positions=8,
        volatility_multiplier=1.0,
        trading_hours="market",
        min_price=1.0,
        max_price=5000.0,
    ),
    AssetType.ETF: AssetConfig(
        asset_type=AssetType.ETF,
        max_position_pct=0.15,
        default_stop_loss_pct=0.05,
        default_take_profit_pct=0.25,
        max_positions=4,
        volatility_multiplier=0.7,
        trading_hours="market",
        min_price=10.0,
        max_price=1000.0,
    ),
    AssetType.CRYPTO: AssetConfig(
        asset_type=AssetType.CRYPTO,
        max_position_pct=0.05,
        default_stop_loss_pct=0.10,
        default_take_profit_pct=0.30,
        max_positions=3,
        volatility_multiplier=2.0,
        trading_hours="24/7",
        min_price=0.01,
        max_price=100000.0,
    ),
}


STOCK_UNIVERSE = [
    "NVDA", "TSLA", "AMD", "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "PLTR", "SOFI", "NIO", "RIVN", "LCID", "MARA", "RIOT", "COIN",
    "SQ", "SHOP", "SNOW", "NET", "CRWD", "DDOG", "ZS", "MDB",
    "RBLX", "U", "HOOD", "AFRM", "UPST", "OPEN",
]

ETF_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE",
    "ARKK", "ARKW", "ARKF", "ARKG",
    "SMH", "SOXX", "XBI", "IBB",
    "GLD", "SLV", "USO", "UNG",
    "TLT", "HYG", "LQD",
    "VXX", "UVXY",
]

CRYPTO_UNIVERSE = [
    "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "MATIC/USD",
    "DOGE/USD", "SHIB/USD", "LTC/USD", "LINK/USD", "UNI/USD",
]

FULL_UNIVERSE: Dict[AssetType, List[str]] = {
    AssetType.STOCK: STOCK_UNIVERSE,
    AssetType.ETF: ETF_UNIVERSE,
    AssetType.CRYPTO: CRYPTO_UNIVERSE,
}


CORRELATION_GROUPS = {
    "mega_tech": ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "QQQ", "XLK", "SMH"],
    "ev_clean": ["TSLA", "RIVN", "LCID", "NIO", "PLUG", "FCEL"],
    "crypto_related": ["MARA", "RIOT", "COIN", "BTC/USD", "ETH/USD"],
    "fintech": ["SQ", "SOFI", "HOOD", "AFRM", "UPST", "XLF"],
    "cloud_saas": ["SNOW", "NET", "CRWD", "DDOG", "ZS", "MDB"],
    "gold": ["GLD", "SLV"],
    "energy": ["XLE", "USO", "UNG"],
    "bonds": ["TLT", "HYG", "LQD"],
    "volatility": ["VXX", "UVXY"],
}


def get_asset_type(symbol: str) -> AssetType:
    """Determine asset type from symbol."""
    if "/" in symbol:
        return AssetType.CRYPTO
    if symbol in ETF_UNIVERSE:
        return AssetType.ETF
    return AssetType.STOCK


def get_asset_config(symbol: str) -> AssetConfig:
    """Get configuration for a symbol based on its asset type."""
    asset_type = get_asset_type(symbol)
    return ASSET_CONFIGS[asset_type]


def get_correlation_group(symbol: str) -> Optional[str]:
    """Find which correlation group a symbol belongs to."""
    for group_name, symbols in CORRELATION_GROUPS.items():
        if symbol in symbols:
            return group_name
    return None


def get_correlated_symbols(symbol: str) -> List[str]:
    """Get all symbols correlated with the given symbol."""
    group = get_correlation_group(symbol)
    if group:
        return [s for s in CORRELATION_GROUPS[group] if s != symbol]
    return []


@dataclass
class PortfolioLimits:
    """Overall portfolio limits."""
    max_total_positions: int = 15
    max_stocks: int = 8
    max_etfs: int = 4
    max_crypto: int = 3
    max_correlated_positions: int = 2
    max_sector_exposure_pct: float = 0.30
    min_cash_reserve_pct: float = 0.05


PORTFOLIO_LIMITS = PortfolioLimits()


def check_position_allowed(
    symbol: str,
    current_positions: List[str],
    limits: PortfolioLimits = PORTFOLIO_LIMITS,
) -> tuple[bool, str]:
    """
    Check if adding a position for this symbol is allowed.
    
    Returns (allowed, reason).
    """
    if len(current_positions) >= limits.max_total_positions:
        return False, f"Max total positions ({limits.max_total_positions}) reached"
    
    if symbol in current_positions:
        return False, f"Already holding {symbol}"
    
    asset_type = get_asset_type(symbol)
    current_by_type = [s for s in current_positions if get_asset_type(s) == asset_type]
    
    type_limits = {
        AssetType.STOCK: limits.max_stocks,
        AssetType.ETF: limits.max_etfs,
        AssetType.CRYPTO: limits.max_crypto,
    }
    
    if len(current_by_type) >= type_limits[asset_type]:
        return False, f"Max {asset_type.value} positions ({type_limits[asset_type]}) reached"
    
    correlated = get_correlated_symbols(symbol)
    correlated_held = [s for s in current_positions if s in correlated]
    
    if len(correlated_held) >= limits.max_correlated_positions:
        return False, f"Max correlated positions ({limits.max_correlated_positions}) for group containing {symbol}"
    
    return True, "Position allowed"
