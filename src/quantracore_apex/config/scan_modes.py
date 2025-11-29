"""
Scan Modes Loader for QuantraCore Apex.

Provides access to pre-defined scanning modes with different
universe coverage, filters, and performance settings.

Version: 9.0-A
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import yaml


SCAN_MODES_CONFIG_PATH = Path("config/scan_modes.yaml")


@dataclass
class FilterConfig:
    """Filter configuration for a scan mode."""
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_atr_percent: Optional[float] = None
    min_relative_volume: Optional[float] = None
    min_avg_volume: Optional[int] = None
    max_float_millions: Optional[float] = None
    min_price_change_5d_percent: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["FilterConfig"]:
        """Create FilterConfig from dictionary."""
        if data is None:
            return None
        return cls(
            min_price=data.get("min_price"),
            max_price=data.get("max_price"),
            min_atr_percent=data.get("min_atr_percent"),
            min_relative_volume=data.get("min_relative_volume"),
            min_avg_volume=data.get("min_avg_volume"),
            max_float_millions=data.get("max_float_millions"),
            min_price_change_5d_percent=data.get("min_price_change_5d_percent"),
        )


@dataclass
class ScanModeConfig:
    """Configuration for a scan mode."""
    name: str
    description: str = ""
    buckets: List[str] = field(default_factory=list)
    max_symbols: int = 1000
    chunk_size: int = 200
    batch_delay_seconds: float = 0.2
    cache_required: bool = True
    priority_sort: Optional[str] = None
    filters: Optional[FilterConfig] = None
    risk_default: str = "medium"
    use_universe: Optional[str] = None
    notes: str = ""
    
    @property
    def is_smallcap_focused(self) -> bool:
        """Check if mode focuses on small caps."""
        smallcap_buckets = {"small", "micro", "nano", "penny"}
        return bool(set(self.buckets) & smallcap_buckets)
    
    @property
    def is_extreme_risk(self) -> bool:
        """Check if mode involves extreme risk assets."""
        return self.risk_default == "extreme" or "nano" in self.buckets or "penny" in self.buckets


@dataclass 
class PerformanceConfig:
    """K6 performance settings."""
    max_concurrent_requests: int = 4
    memory_limit_mb: int = 2048
    cpu_throttle_percent: int = 70
    disk_cache_max_gb: int = 10
    batch_cooldown_seconds: float = 1.0


@dataclass
class ScanModesRegistry:
    """Registry of all scan modes."""
    version: str = "9.0-A"
    modes: Dict[str, ScanModeConfig] = field(default_factory=dict)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    data_providers: Dict[str, str] = field(default_factory=dict)


_modes_cache: Optional[ScanModesRegistry] = None


def _load_config() -> dict:
    """Load raw YAML configuration."""
    config_path = SCAN_MODES_CONFIG_PATH
    if not config_path.exists():
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "scan_modes.yaml"
    
    if not config_path.exists():
        return {"version": "9.0-A", "modes": {}}
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def _load_registry(force_reload: bool = False) -> ScanModesRegistry:
    """Load the scan modes registry."""
    global _modes_cache
    
    if _modes_cache is not None and not force_reload:
        return _modes_cache
    
    raw_config = _load_config()
    
    registry = ScanModesRegistry(
        version=raw_config.get("version", "9.0-A"),
    )
    
    modes_raw = raw_config.get("modes", {})
    for mode_name, mode_data in modes_raw.items():
        if not isinstance(mode_data, dict):
            continue
        
        filters = FilterConfig.from_dict(mode_data.get("filters"))
        
        mode_config = ScanModeConfig(
            name=mode_name,
            description=mode_data.get("description", ""),
            buckets=mode_data.get("buckets", []),
            max_symbols=mode_data.get("max_symbols", 1000),
            chunk_size=mode_data.get("chunk_size", 200),
            batch_delay_seconds=mode_data.get("batch_delay_seconds", 0.2),
            cache_required=mode_data.get("cache_required", True),
            priority_sort=mode_data.get("priority_sort"),
            filters=filters,
            risk_default=mode_data.get("risk_default", "medium"),
            use_universe=mode_data.get("use_universe"),
            notes=mode_data.get("notes", ""),
        )
        registry.modes[mode_name] = mode_config
    
    perf_raw = raw_config.get("performance", {}).get("k6_safe_defaults", {})
    if perf_raw:
        registry.performance = PerformanceConfig(
            max_concurrent_requests=perf_raw.get("max_concurrent_requests", 4),
            memory_limit_mb=perf_raw.get("memory_limit_mb", 2048),
            cpu_throttle_percent=perf_raw.get("cpu_throttle_percent", 70),
            disk_cache_max_gb=perf_raw.get("disk_cache_max_gb", 10),
            batch_cooldown_seconds=perf_raw.get("batch_cooldown_seconds", 1.0),
        )
    
    registry.data_providers = raw_config.get("data_providers", {
        "primary": "polygon",
        "secondary": "yahoo",
        "tertiary": "csv_bundle",
    })
    
    _modes_cache = registry
    return registry


def load_scan_mode(name: str) -> ScanModeConfig:
    """
    Load a scan mode configuration by name.
    
    Args:
        name: Mode name (e.g., "full_us_equities", "high_vol_small_caps")
        
    Returns:
        ScanModeConfig for the mode
        
    Raises:
        ValueError: If mode not found
    """
    registry = _load_registry()
    
    if name not in registry.modes:
        available = list(registry.modes.keys())
        raise ValueError(f"Unknown scan mode: {name}. Available: {available}")
    
    return registry.modes[name]


def list_scan_modes() -> List[str]:
    """
    List all available scan modes.
    
    Returns:
        List of mode names
    """
    registry = _load_registry()
    return list(registry.modes.keys())


def get_scan_mode_config(name: str) -> ScanModeConfig:
    """
    Alias for load_scan_mode.
    
    Args:
        name: Mode name
        
    Returns:
        ScanModeConfig
    """
    return load_scan_mode(name)


def get_performance_config() -> PerformanceConfig:
    """
    Get K6 performance configuration.
    
    Returns:
        PerformanceConfig with K6-safe defaults
    """
    registry = _load_registry()
    return registry.performance


def get_data_provider_priority() -> Dict[str, str]:
    """
    Get data provider priority configuration.
    
    Returns:
        Dict with primary, secondary, tertiary providers
    """
    registry = _load_registry()
    return registry.data_providers


def get_default_mode() -> str:
    """
    Get the default scan mode name.
    
    Returns:
        Default mode name (mega_large_focus for safety)
    """
    return "mega_large_focus"


def get_smallcap_modes() -> List[str]:
    """
    Get all modes that focus on small caps.
    
    Returns:
        List of small-cap focused mode names
    """
    registry = _load_registry()
    return [
        name for name, mode in registry.modes.items()
        if mode.is_smallcap_focused
    ]


def get_extreme_risk_modes() -> List[str]:
    """
    Get all modes that involve extreme risk assets.
    
    Returns:
        List of extreme-risk mode names
    """
    registry = _load_registry()
    return [
        name for name, mode in registry.modes.items()
        if mode.is_extreme_risk
    ]
